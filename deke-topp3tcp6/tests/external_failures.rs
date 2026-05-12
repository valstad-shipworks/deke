//! Boilerplate for triaging failing trajectories pulled in from external projects.
//!
//! ## How to add a failing trajectory
//!
//! 1. Copy a `#[test]` from the bottom of this file (e.g. `traj_template`) into a new
//!    test function. Rename it after the failure (e.g. `traj_2025_05_11_factorization_fail`).
//! 2. Paste the waypoint list into `WAYPOINTS`.
//! 3. Tweak the constraints (joint/TCP limits, boundary, sample_rate_hz, locked_prefix)
//!    to match the producing project's settings.
//! 4. Run with `cargo test --test external_failures -- --nocapture <test_name>` to see
//!    the full diagnostic dump.
//!
//! The test wrapper (`run_external_traj`) prints the diagnostic *whether the retime
//! succeeds or fails* — that's the point. Use the diagnostic to identify which suspect
//! mechanism the failure matches:
//!
//! - **PCHIP overshoot** → `derivative_stats.max_abs_qpp` / `max_abs_qppp` huge,
//!   localized to a specific (sample, joint).
//! - **Relative-qp cutoff over-removing** → `derivative_stats.degenerate_qp_samples > 0`
//!   or `min_qp_norm_relative_sq` near `1e-6`.
//! - **TCP scaling collapse** → one entry of `tcp_stats.min_abs_pp_per_axis` near zero
//!   while another is O(1).
//! - **Initial guess** → `initial_guess.end_sdd_residual` huge, or
//!   `boundary_slack_usage.end_sdd` near `boundary_slack`.
//! - **Path geometry** → `path_stats.segment_length_ratio` huge.

use deke_topp3tcp6::nlp::{build_and_solve, build_and_solve_warm};
use deke_topp3tcp6::path_derivatives::PathDerivatives;
use deke_topp3tcp6::{
    SolveStatus, TcpLimits, Topp3Tcp6, Topp3Tcp6Constraints,
};
use deke_topp3tcp6::boundary;
use deke_types::{JointValidator, Retimer, SRobotPath, SRobotQ, URDFChain, URDFJoint};

// ----------------------------------------------------------------------------
// FK chain — copied verbatim from the user's other project
// ----------------------------------------------------------------------------

const URDF_JOINTS: [URDFJoint; 6] = [
    URDFJoint::revolute(
        (0f64, 0f64, 0.152f64),
        (0f64, -0f64, 0f64),
        (0f64, 0f64, 1f64),
    ),
    URDFJoint::revolute(
        (0.075f64, -0.105f64, 0.273f64),
        (-1.5708f64, -0f64, 0f64),
        (0f64, 0f64, 1f64),
    ),
    URDFJoint::revolute(
        (-0.00000000000000625888f64, -0.84f64, 0.04028f64),
        (-3.14159f64, -0.0000000000000000252315f64, -0.00000000000000423966f64),
        (0f64, 0f64, 1f64),
    ),
    URDFJoint::revolute(
        (0.295618f64, 0.215f64, -0.0642f64),
        (-3.14159f64, 1.5708f64, 0f64),
        (0f64, 0f64, 1f64),
    ),
    URDFJoint::revolute(
        (-0.0501976f64, 0.000491285f64, -1.0445f64),
        (-3.13181f64, 1.5708f64, 0f64),
        (0f64, 0f64, 1f64),
    ),
    URDFJoint::revolute(
        (0.075182f64, -0.00000000205208f64, -0.0507f64),
        (3.14159f64, 1.5708f64, 0f64),
        (0f64, 0f64, 1f64),
    ),
];

const URDF_FIXED_SUFFIX: [URDFJoint; 1] = [URDFJoint::fixed(
    (
        -0.00000000000000056205f64,
        -0.000000000000000888178f64,
        -0.00000000000000310862f64,
    ),
    (-1.5708f64, 1.5708f64, 3.14159f64),
)];

/// Builds the user's external 6-DOF URDF chain in `f64`. Mirrors the construction in the
/// producing project; if their chain ever changes, update both `URDF_JOINTS` and
/// `URDF_FIXED_SUFFIX` above to match.
fn external_chain() -> URDFChain<6, f64> {
    match URDFChain::<6, f64>::new_f64(URDF_JOINTS) {
        Ok(c) => match c.with_fixed_suffix_f64(&URDF_FIXED_SUFFIX) {
            Ok(c2) => c2,
            Err(e) => panic!(
                "URDF fixed-suffix attach failed (this is a fixture bug, not a retimer bug): {:?}",
                e
            ),
        },
        Err(e) => panic!(
            "URDFChain::new_f64 failed on the external joints (fixture bug): {:?}",
            e
        ),
    }
}

fn wide_validator() -> JointValidator<6, f64> {
    JointValidator::<6, f64>::new(
        SRobotQ::from_array([-10.0; 6]),
        SRobotQ::from_array([10.0; 6]),
    )
}

/// The exact constraints used by the producing project. Per-joint v/a/j limits, TCP
/// limits, sample rate, boundary, etc. all match the log lines pasted in by the user.
fn external_cfg() -> Topp3Tcp6Constraints<6> {
    use deke_topp3tcp6::JointLimits;
    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.0, 1.0, 1.0); // gets overwritten
    cfg.joint = JointLimits {
        q_min: SRobotQ::from_array([f64::NEG_INFINITY; 6]),
        q_max: SRobotQ::from_array([f64::INFINITY; 6]),
        v_max: SRobotQ::from_array([
            2.748894, 2.748894, 3.468842, 5.497787, 5.890486, 9.424778,
        ]),
        a_max: SRobotQ::from_array([
            6.170564, 6.170564, 7.786665, 12.341129, 13.222637, 21.156220,
        ]),
        j_max: SRobotQ::from_array([
            22.897033, 22.897033, 28.893875, 45.794066, 49.065074, 78.504119,
        ]),
    };
    cfg.tcp = Some(TcpLimits {
        v_max: 2.0,
        a_max: 20.0,
        j_max: 200.0,
    });
    cfg.sample_rate_hz = 125.0;
    cfg.post_validation = false;
    // Match the producing project's looser projection tolerance (default is 1e-4).
    cfg.boundary.projection_tolerance = 1e-3;
    cfg
}

// ----------------------------------------------------------------------------
// Wrapper — runs the retime, prints the full diagnostic, returns success bool
// ----------------------------------------------------------------------------

/// Runs `Topp3Tcp6.retime` against the external URDF chain and dumps the diagnostic to
/// stderr regardless of outcome. Returns the success boolean — the caller can `assert!`
/// or not depending on what it wants from the test.
fn run_external_traj(
    name: &str,
    waypoints: Vec<SRobotQ<6, f64>>,
    cfg: Topp3Tcp6Constraints<6>,
) -> bool {
    let fk = external_chain();
    let path = match SRobotPath::<6, f64>::try_new(waypoints) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[{name}] path constructor rejected: {e}");
            return false;
        }
    };
    let mut validator = wide_validator();
    let (result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
    eprintln!("[{name}] diagnostic:\n{diag}");
    if let Err(e) = &result {
        eprintln!("[{name}] retime returned error: {e}");
    }
    result.is_ok()
}

// ----------------------------------------------------------------------------
// Tests — paste new failing trajectories here
// ----------------------------------------------------------------------------

/// Smoke test: a healthy single-segment retime against the external chain, using the
/// same `external_cfg()` as the rest of this file. Confirms the fixture is sane and
/// stays consistent with the regime the captured-trajectory tests run in. If this
/// fails, the chain construction is wrong, not the trajectory.
#[test]
fn smoke_external_chain_works() {
    let waypoints = vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.3, -0.7, 0.9, 0.2, 0.1, 0.3]),
    ];
    let ok = run_external_traj("smoke_external_chain", waypoints, external_cfg());
    assert!(ok, "external-chain smoke test failed; the fixture is broken");
}

/// Template — copy this into a new test function and paste your failing trajectory in
/// place of the `WAYPOINTS` and `cfg` below. Remove the `#[ignore]` once you have a real
/// trajectory loaded; the placeholder data here is intentionally trivial.
#[test]
#[ignore = "template — copy and paste a real failing trajectory in"]
fn traj_template() {
    let waypoints = vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
    ];

    let mut cfg = Topp3Tcp6Constraints::<6>::symmetric(1.5, 5.0, 200.0);
    cfg.tcp = Some(TcpLimits {
        v_max: f64::INFINITY,
        a_max: f64::INFINITY,
        j_max: f64::INFINITY,
    });
    cfg.sample_rate_hz = 125.0;
    // cfg.boundary = ...;
    // cfg.locked_prefix = 0;
    // cfg.solver.max_iterations = ...;

    let ok = run_external_traj("traj_template", waypoints, cfg);
    // Drop the assert if you want the test to pass-on-failure (e.g. for diagnostic
    // gathering only); keep it if you want CI to flag regressions.
    let _ = ok;
}

/// Convenience: the [`SolveStatus`] enum is re-exported so test asserts can match on
/// specific failure modes.
#[allow(dead_code)]
fn _solve_status_is_re_exported(_s: SolveStatus) {}

// ----------------------------------------------------------------------------
// Failing trajectories pulled from the user's project (2025-11-11)
//
// Each test reproduces a single warning line from `valstad::controls::universal::traj_gen`.
// All share `external_cfg()` (TCP v=2 a=20 j=200, per-joint limits scaled to the
// producing chain, rest-to-rest, default solver options). The asserts here are
// `assert!(ok)` so CI flags any test that *starts* passing — that's a fix to celebrate.
// ----------------------------------------------------------------------------

/// 2 waypoints, feasibility restoration failed in 66 iter / 37 ms.
#[test]
fn external_2wp_feas_restore() {
    let waypoints = vec![
        SRobotQ::from_array([
            -1.024476975265872, 0.685236702990971, -1.094963611932775,
            -1.1018032138195069, -1.628005003397021, -2.5771111826649475,
        ]),
        SRobotQ::from_array([
            -1.120997183009307, 0.6357904873823756, -1.1284028943166988,
            -1.0116608901465864, -1.6476318609769856, -2.5273633150398234,
        ]),
    ];
    let ok = run_external_traj("external_2wp_feas_restore", waypoints, external_cfg());
    assert!(ok, "still failing — check diagnostic above");
}

/// 13 waypoints, feasibility restoration failed in 16 iter / 136 ms.
#[test]
fn external_13wp_feas_restore() {
    let waypoints = vec![
        SRobotQ::from_array([-0.6120284630576525, 0.6487779263898087, -1.0738133155393088, -0.32136803612032755, 0.8516805909536622, 2.3894897592935407]),
        SRobotQ::from_array([-0.6338046599965044, 0.5810911315504893, -0.9726977537492854, -0.043393784502349604, 0.46170146939218554, 1.7064995445466689]),
        SRobotQ::from_array([-0.6449861446481061, 0.5456296282910937, -0.9187692022095447, 0.12494356908084228, 0.21513661999451988, 1.3153094953054927]),
        SRobotQ::from_array([-0.6535737492721188, 0.5176269441270557, -0.8757384262798767, 0.3760705889966836, -0.055515321661715876, 0.8720639518396054]),
        SRobotQ::from_array([-0.6575249515021748, 0.5040597192793315, -0.8546586119975126, 0.47792698231589614, -0.2654213566385266, 0.6606152688943149]),
        SRobotQ::from_array([-0.6583544602921314, 0.5007342471814625, -0.8494652962564302, 0.563573495430691, -0.3978477831086992, 0.5318157151701427]),
        SRobotQ::from_array([-0.6583491092653706, 0.5004768572606678, -0.8490659222306935, 0.5991873720086153, -0.4525099739101317, 0.48381922604605665]),
        SRobotQ::from_array([-0.6580613535380896, 0.5008929044245669, -0.8497241392456283, 0.6646719886901518, -0.5527240584305901, 0.3997596847158693]),
        SRobotQ::from_array([-0.6577165028269327, 0.5022145047243908, -0.852602218501489, 0.7085183723497458, -0.6175620225456778, 0.33860275849277005]),
        SRobotQ::from_array([-0.657078209209505, 0.5055524758876971, -0.860259933984505, 0.7584272617124274, -0.6875094730322684, 0.2595443611393028]),
        SRobotQ::from_array([-0.655506806250084, 0.5145929593748413, -0.8815686613478313, 0.8790955219358413, -0.852673275036928, 0.05761494192171879]),
        SRobotQ::from_array([-0.6536967400501428, 0.5509831070147456, -0.9866038107240835, 1.2366406639612897, -1.26355114158666, -0.8189383752067887]),
        SRobotQ::from_array([-0.6571856831702497, 0.5881793596638587, -1.1189216441219967, 1.4475834177344886, -1.3713763072450946, -1.8164736747204508]),
    ];
    let ok = run_external_traj("external_13wp_feas_restore", waypoints, external_cfg());
    assert!(ok, "still failing — check diagnostic above");
}

/// 20 waypoints, feasibility restoration failed in 62 iter / 370 ms.
#[test]
fn external_20wp_feas_restore() {
    let waypoints = vec![
        SRobotQ::from_array([-0.13178508229116317, 0.9075342514033721, -0.9188427013284615, 0.7969249820991294, -1.819674932175745, -0.49626522856385724]),
        SRobotQ::from_array([-0.21276625706009744, 0.8526692070768522, -0.8539050776956741, 0.5469984503351912, -1.742603616652607, -0.28241182019117644]),
        SRobotQ::from_array([-0.30302824320417415, 0.7848675275829271, -0.8011141547937695, 0.3136880991820056, -1.6306727167576507, -0.07327575788355999]),
        SRobotQ::from_array([-0.5704146847719728, 0.5469855939762924, -0.7520901374966994, -0.12762301742912607, -1.1542108031556406, 0.38665774040301437]),
        SRobotQ::from_array([-0.6917605716682411, 0.4292917030211627, -0.7570645297852358, -0.27456020860405994, -0.8857215938006072, 0.5693445050001721]),
        SRobotQ::from_array([-0.7088855404246883, 0.40149573466354793, -0.7845300365694592, -0.30168309934502124, -0.7085994383990989, 0.654639426465333]),
        SRobotQ::from_array([-0.7123415747381341, 0.39173178471358217, -0.8001229916166017, -0.30962618218099586, -0.6203673706744983, 0.6942903039834469]),
        SRobotQ::from_array([-0.715439045982729, 0.3825311562652645, -0.815172679111025, -0.31788314284820723, -0.534687097312084, 0.7322083930513807]),
        SRobotQ::from_array([-0.717838908656336, 0.3744051757750931, -0.8291127368742197, -0.32686867756390087, -0.4541396081806532, 0.7667816695838424]),
        SRobotQ::from_array([-0.7197108300541879, 0.3670944412100236, -0.8422227457799604, -0.3362444027529109, -0.37743947864099964, 0.7986860841755581]),
        SRobotQ::from_array([-0.7222161048965842, 0.3543965911094322, -0.8665080670372686, -0.3546031704160479, -0.2330154480203029, 0.8549588638143472]),
        SRobotQ::from_array([-0.7213653012887886, 0.3423709060111504, -0.8955221025270905, -0.34933131142556545, -0.05053739211135967, 0.8795922983789021]),
        SRobotQ::from_array([-0.7184855585775798, 0.3378835509676265, -0.9111028425784191, -0.47800224931866797, 0.054195081615297964, 1.014634440185975]),
        SRobotQ::from_array([-0.7166852818469561, 0.33618259671113404, -0.9183412915610317, -0.47421241467829556, 0.10385903175849864, 1.0119829123982254]),
        SRobotQ::from_array([-0.714537613585539, 0.3350077794159263, -0.9250132137106487, -0.4830974826812482, 0.15096804009514783, 1.0199702578370133]),
        SRobotQ::from_array([-0.7118918352287608, 0.3338927816741182, -0.9311553501291223, -0.5019067762602264, 0.19396459019135132, 1.0296928425663647]),
        SRobotQ::from_array([-0.708255999766908, 0.33286349860054026, -0.9362306570734298, -0.5352198053938696, 0.22871881074747402, 1.0375507686123397]),
        SRobotQ::from_array([-0.6995752607672129, 0.3311364172893025, -0.9453325222498303, -0.6165714102047681, 0.2878168577237908, 1.051424989533304]),
        SRobotQ::from_array([-0.6266200482058061, 0.3396376696789162, -0.9931962220620943, -1.0497219341223591, 0.42120317356073866, 1.134072732427895]),
        SRobotQ::from_array([-0.5958628658454487, 0.3472965126707271, -1.0083537428158458, -1.1861225056698494, 0.41158354147440424, 1.1614944953811925]),
    ];
    let ok = run_external_traj("external_20wp_feas_restore", waypoints, external_cfg());
    assert!(ok, "still failing — check diagnostic above");
}

/// 8 waypoints, feasibility restoration failed in 73 iter / 475 ms.
#[test]
fn external_8wp_feas_restore() {
    let waypoints = vec![
        SRobotQ::from_array([-0.6364632544213349, 0.3802690502211359, -0.9972162375620871, -1.1558150171008872, 0.3874953228516746, 1.1660525739447671]),
        SRobotQ::from_array([-0.721846280710852, 0.37576783004126607, -0.9307956849654027, -0.7378766433380558, 0.3050033760616249, 1.1802386380717917]),
        SRobotQ::from_array([-0.7504928867152842, 0.3708009486130261, -0.9006694236872432, -0.5975635473851985, 0.27467197128185794, 1.1677764895126956]),
        SRobotQ::from_array([-0.7502668817422785, 0.3626875882640009, -0.8878925771004293, -0.5771302319628522, 0.2635020161312982, 1.118628367931161]),
        SRobotQ::from_array([-0.6057486690247245, 0.17619503374059908, -0.864866663660868, 0.39110067478216165, -0.16813003301110035, -0.8783368611747007]),
        SRobotQ::from_array([-0.5815544282609763, 0.15464996057632047, -0.8760812096089616, 0.5296678139464301, -0.22625197515831497, -1.1552106572603036]),
        SRobotQ::from_array([-0.5572599641132697, 0.13311599806229463, -0.8873891827746738, 0.668517912042139, -0.28417834246355345, -1.4324307081889254]),
        SRobotQ::from_array([-0.4634180866210337, 0.25899651320682565, -0.9699935892427178, 1.158330849209562, -0.44912260485450173, -1.582155970300871]),
    ];
    let ok = run_external_traj("external_8wp_feas_restore", waypoints, external_cfg());
    assert!(ok, "still failing — check diagnostic above");
}

/// 11 waypoints, feasibility restoration failed in 55 iter / 549 ms.
#[test]
fn external_11wp_feas_restore() {
    let waypoints = vec![
        SRobotQ::from_array([-0.5928746322636275, 0.6291960713825318, -1.0844215473821865, -0.34388315508210915, 0.8630210997030802, 2.395735974357877]),
        SRobotQ::from_array([-0.570847741637278, 0.5468212860384606, -0.9984551071214725, -0.1367467514947923, 0.5111366225535745, 1.743124409740365]),
        SRobotQ::from_array([-0.560823026441177, 0.5071897554563674, -0.9595185918911131, -0.03097634809713723, 0.3222614789040163, 1.4084807375390105]),
        SRobotQ::from_array([-0.5494029167312134, 0.4570076456950763, -0.9154637377433994, 0.08549006088626229, 0.046922774723230684, 0.9754146920492083]),
        SRobotQ::from_array([-0.5415729297886339, 0.4114374810027875, -0.8847656497509746, 0.30633839833281323, -0.27029284191673764, 0.4133717567557664]),
        SRobotQ::from_array([-0.540858224383945, 0.39229798450891457, -0.8806736773501005, 0.40425942924476604, -0.46394242735611363, 0.12403351927004845]),
        SRobotQ::from_array([-0.5407270887560754, 0.3742312839953723, -0.8793896084079533, 0.5078459535331665, -0.6649801394178785, -0.17583301080149555]),
        SRobotQ::from_array([-0.54887376569767, 0.3665180591134653, -0.8972758823377971, 0.6893667274747418, -0.9359711569710448, -0.6247728041322441]),
        SRobotQ::from_array([-0.5801273382772537, 0.41063509719451124, -0.9675152168629471, 0.9358890671320554, -1.1198269466470956, -1.058531258065017]),
        SRobotQ::from_array([-0.645127042756434, 0.5141752488097472, -1.112774752411208, 1.3692845976827532, -1.3507290355089117, -1.734442879616071]),
        SRobotQ::from_array([-0.6602634083150769, 0.5413226986305902, -1.1454365501248789, 1.4542001514201204, -1.3665700681397834, -1.8416616992126222]),
    ];
    let ok = run_external_traj("external_11wp_feas_restore", waypoints, external_cfg());
    assert!(ok, "still failing — check diagnostic above");
}

/// 6 waypoints, **factorization failed** in 324 iter / 2.4 s. This is the suspected
/// SOC-needs-Sleipnir-side-fix bucket — diagnostic should reveal which trigger fired.
#[test]
fn external_6wp_factorization_fail() {
    let waypoints = vec![
        SRobotQ::from_array([0.010423163653220588, 0.3036922098121164, -1.4743607380023693, -1.4408071324378127, 1.559611068218138, 1.7651037673642795]),
        SRobotQ::from_array([-0.10230958396921962, 0.16009533412481336, -1.0977035299095348, -1.3408354873741761, 0.9943244272351427, 0.30716826349078813]),
        SRobotQ::from_array([-0.22362521900689952, 0.19830394521315076, -1.0250815140117073, -1.343887121658254, 0.5858985186687447, -0.042935631885345486]),
        SRobotQ::from_array([-0.3114168845087661, 0.24180779799961907, -0.9980330100420046, -1.3547900923140528, 0.3034694659526835, -0.21916323127210138]),
        SRobotQ::from_array([-0.6801815415312936, 0.531994726399778, -0.9725025698070294, -1.4490332695138926, -1.0427254283274527, -1.779545445845086]),
        SRobotQ::from_array([-0.6810699673901779, 0.6383279529509278, -1.0504101612409238, -1.4985658170207112, -1.2572478387358357, -2.8937194892592473]),
    ];
    let ok = run_external_traj("external_6wp_factorization_fail", waypoints, external_cfg());
    assert!(ok, "still failing — check diagnostic above");
}

/// 5 waypoints, locally infeasible (JointVelocity) in 305 iter / 2.87 s.
#[test]
fn external_5wp_locally_infeasible() {
    let waypoints = vec![
        SRobotQ::from_array([-0.2600530969913884, 0.5107495417579586, -1.1968085675397409, -0.7838563915925326, 1.1227049681102683, 2.3930946752730757]),
        SRobotQ::from_array([-0.3002753240296815, 0.45366120097426194, -0.9774216892107865, -0.9470203310825394, 0.6009268509250804, 1.0592893141164614]),
        SRobotQ::from_array([-0.6625843873425555, 0.5298660403525086, -0.8565395158403143, -1.2208570407996082, -0.7402167089284234, -1.2632068554889855]),
        SRobotQ::from_array([-0.799900286456148, 0.7609017444406853, -0.966903575164756, -1.3752151479025385, -1.3300551355303485, -2.945139688420589]),
        SRobotQ::from_array([-0.7996621311771062, 0.7672513254384694, -0.9717101591712993, -1.3775498245596665, -1.330178335367621, -2.9668419986180865]),
    ];
    let ok = run_external_traj("external_5wp_locally_infeasible", waypoints, external_cfg());
    assert!(ok, "still failing — check diagnostic above");
}

/// 3 waypoints, max iterations exceeded after 1876 iter / 5 s. The waypoint count is
/// small but the IPM ground for a long time — likely a feasibility-restoration loop.
#[test]
fn external_3wp_max_iter() {
    let waypoints = vec![
        SRobotQ::from_array([-0.4428234938052882, 0.8248214898114147, -0.9093426911981417, 1.1244239220008716, -1.4317094551832892, -0.13325750467285322]),
        SRobotQ::from_array([-0.4462468406644217, 0.6783644898450626, -0.8532621186840782, 0.6950203781161692, -1.103606256052987, 0.5611015295767559]),
        SRobotQ::from_array([-0.5214724468931752, 0.43630629312954655, -0.856249087693265, 0.05309342439974587, -0.6693423042103546, 1.986488815847895]),
    ];
    let ok = run_external_traj("external_3wp_max_iter", waypoints, external_cfg());
    assert!(ok, "still failing — check diagnostic above");
}

/// 10 waypoints, max iterations exceeded after 2210 iter / 13.5 s. Notably similar in
/// shape to `external_3wp_max_iter` (joint 5 sweeping +/- 3 rad) but with more
/// intermediate samples.
#[test]
fn external_10wp_max_iter() {
    let waypoints = vec![
        SRobotQ::from_array([-0.015416748398413367, 0.3316967825001088, -1.4531241229739642, -1.414453544912589, 1.5547182921717306, 1.7858693392183467]),
        SRobotQ::from_array([-0.028872212555163076, 0.3218519406951016, -1.3440183202887233, -1.3930009347468584, 1.390237939799394, 1.281676115388829]),
        SRobotQ::from_array([-0.04918673746968716, 0.31748902008809005, -1.2732208163878307, -1.3807028446592762, 1.2377716292031853, 0.9383346612705158]),
        SRobotQ::from_array([-0.10450329816659168, 0.31714547973284385, -1.1872065317540315, -1.3698717747523952, 0.9350578132676379, 0.48095698189449926]),
        SRobotQ::from_array([-0.2180046093864924, 0.3221143567084731, -1.0364455645111756, -1.3516909915914341, 0.34154396312188334, -0.3528378432081694]),
        SRobotQ::from_array([-0.22684049498503916, 0.32169121864795414, -1.028501883693815, -1.3513074035618147, 0.30221313765402635, -0.3973428365185442]),
        SRobotQ::from_array([-0.3607170841501045, 0.3283709003798248, -0.9517984078266251, -1.3712655837701617, -0.22408956656963475, -0.9616627055646206]),
        SRobotQ::from_array([-0.46915518715953525, 0.3439046861562589, -0.91510399525184, -1.388446570148907, -0.6125121067364329, -1.379485802537151]),
        SRobotQ::from_array([-0.5692940772276867, 0.4261348098912236, -0.9334502532006363, -1.4216148787472895, -0.9334758138527821, -1.9275121142798881]),
        SRobotQ::from_array([-0.6911968393325083, 0.6460369344365877, -1.0452777015839356, -1.4874069589458077, -1.2624125173223177, -2.898185959243144]),
    ];
    let ok = run_external_traj("external_10wp_max_iter", waypoints, external_cfg());
    assert!(ok, "still failing — check diagnostic above");
}

/// Captured from `external_bench` (seed `0xC0FFEE_DECAF_F00D`, p2p-tiny run 1/8). A
/// 2-waypoint path with `Δ ≈ 0.05 rad` per joint that the IPM declares
/// `LocallyInfeasible` after 158 iter. Worth checking the diagnostic to see whether the
/// PCHIP derivatives go bad at this near-singular configuration.
#[test]
fn bench_p2p_tiny_locally_infeasible() {
    let waypoints = vec![
        SRobotQ::from_array([1.0885119, 0.7386001, 0.4530678, 1.4205396, 0.1884099, 1.7257718]),
        SRobotQ::from_array([1.0519463, 0.7447735, 0.5023250, 1.4372716, 0.1531237, 1.7296510]),
    ];
    let ok = run_external_traj("bench_p2p_tiny", waypoints, external_cfg());
    assert!(ok, "still failing — check diagnostic above");
}

/// Bench p2p-small run 4/8 — same failure shape, 2 wp, `Δ ≈ 0.2 rad`, 318 iter.
#[test]
fn bench_p2p_small_locally_infeasible() {
    let waypoints = vec![
        SRobotQ::from_array([-1.1703483, -0.3139854, 0.1235649, -1.0546926, 0.1807552, -0.9767784]),
        SRobotQ::from_array([-1.3687564, -0.4719839, 0.3012058, -1.1227008, 0.1640270, -0.9921301]),
    ];
    let ok = run_external_traj("bench_p2p_small", waypoints, external_cfg());
    assert!(ok, "still failing — check diagnostic above");
}

/// Bench p2p-medium run 4/8 — 2 wp, `Δ ≈ 0.6 rad`, 165 iter.
#[test]
fn bench_p2p_medium_locally_infeasible() {
    let waypoints = vec![
        SRobotQ::from_array([-0.6496428, -0.4632142, -0.6906568, 0.6976452, 0.1852536, 0.0864369]),
        SRobotQ::from_array([-0.9760553, -0.5811528, -0.3409180, 1.1595234, 0.4127398, 0.4162376]),
    ];
    let ok = run_external_traj("bench_p2p_medium", waypoints, external_cfg());
    assert!(ok, "still failing — check diagnostic above");
}

/// Determinism check: runs the same 50-waypoint trajectory 5 times in a single test
/// process and asserts every diagnostic field matches bit-for-bit (apart from wall-clock
/// timings, which are inherently jittery). If this fails it means Sleipnir's IPM is
/// non-deterministic *within a single binary* — separate from the cross-opt-level
/// numerical drift we already documented.
///
/// Run with `cargo test --release ...` to also probe the FMA-enabled path.
#[test]
fn bench_multi_seg_50wp_is_deterministic_within_binary() {
    let waypoints = bench_multi_seg_50wp_waypoints();
    let fk = external_chain();
    let cfg = external_cfg();
    let mut validator = wide_validator();

    let mut prev = None::<deke_topp3tcp6::Topp3Tcp6Diagnostic>;
    for run in 0..5 {
        let path = SRobotPath::<6, f64>::try_new(waypoints.clone()).unwrap();
        let (_result, diag) = Topp3Tcp6.retime(&cfg, &path, &fk, &mut validator, &());
        eprintln!(
            "run {}: status={:?} iter={} sddd_init_max={:.6e} slack_sd0={:.6e}",
            run,
            diag.status,
            diag.iterations,
            diag.initial_guess.max_sddd,
            diag.boundary_slack_usage.start_sd,
        );
        if let Some(p) = &prev {
            assert_eq!(
                diag.status, p.status,
                "run {} status diverged from run 0",
                run
            );
            assert_eq!(
                diag.iterations, p.iterations,
                "run {} iter count diverged from run 0",
                run
            );
            assert_eq!(
                diag.densified_samples, p.densified_samples,
                "densified count diverged",
            );
            assert_eq!(
                diag.output_samples, p.output_samples,
                "output count diverged",
            );
            // Compare a handful of key f64 fields bit-exactly. Wall times are excluded
            // because they're noisy by definition.
            assert_eq!(
                diag.peak_joint_velocity.to_bits(),
                p.peak_joint_velocity.to_bits(),
                "peak_joint_velocity diverged",
            );
            assert_eq!(
                diag.peak_tcp_acceleration.to_bits(),
                p.peak_tcp_acceleration.to_bits(),
                "peak_tcp_acceleration diverged",
            );
            assert_eq!(
                diag.derivative_stats.max_abs_qpp.to_bits(),
                p.derivative_stats.max_abs_qpp.to_bits(),
                "max_abs_qpp diverged",
            );
            assert_eq!(
                diag.initial_guess.max_sddd.to_bits(),
                p.initial_guess.max_sddd.to_bits(),
                "initial_guess.max_sddd diverged",
            );
            assert_eq!(
                diag.boundary_slack_usage.start_sd.to_bits(),
                p.boundary_slack_usage.start_sd.to_bits(),
                "boundary_slack_usage.start_sd diverged",
            );
        }
        prev = Some(diag);
    }
}

fn bench_multi_seg_50wp_waypoints() -> Vec<SRobotQ<6, f64>> {
    vec![
        SRobotQ::from_array([0.3343222, 0.8929453, -0.0536905, 0.7182915, -1.2274445, -1.3873202]),
        SRobotQ::from_array([0.2674488, 0.9586826, -0.0377492, 0.6382554, -1.2414511, -1.3031950]),
        SRobotQ::from_array([0.3439960, 0.9346094, -0.1304580, 0.6939549, -1.2237945, -1.3037018]),
        SRobotQ::from_array([0.3770593, 0.8371893, -0.0471791, 0.7772376, -1.2808111, -1.3488145]),
        SRobotQ::from_array([0.3357331, 0.8777153, -0.0870164, 0.8393403, -1.2695598, -1.3989979]),
        SRobotQ::from_array([0.4185733, 0.9508877, -0.0808071, 0.9095306, -1.2537831, -1.3341339]),
        SRobotQ::from_array([0.4889850, 0.8706944, -0.0575931, 0.9365597, -1.3327671, -1.2880048]),
        SRobotQ::from_array([0.5089944, 0.7954211, -0.0015010, 0.9421552, -1.3384181, -1.1912302]),
        SRobotQ::from_array([0.5669821, 0.8019158, 0.0249302, 0.9983462, -1.2633414, -1.2682264]),
        SRobotQ::from_array([0.5462322, 0.8269755, 0.0114744, 1.0248954, -1.2638821, -1.1814715]),
        SRobotQ::from_array([0.4478502, 0.8240671, -0.0374530, 1.0970184, -1.2539433, -1.2531433]),
        SRobotQ::from_array([0.3921103, 0.8241266, 0.0245878, 1.1726546, -1.1683801, -1.3131421]),
        SRobotQ::from_array([0.2961473, 0.8801921, 0.0635315, 1.1099756, -1.1120828, -1.3019841]),
        SRobotQ::from_array([0.3483047, 0.8519227, 0.0110124, 1.0240691, -1.0440697, -1.2836198]),
        SRobotQ::from_array([0.2576151, 0.8472690, 0.0692917, 0.9619144, -1.0929707, -1.3585659]),
        SRobotQ::from_array([0.2756267, 0.8216234, 0.0963444, 0.9842113, -1.1360959, -1.3027613]),
        SRobotQ::from_array([0.3094045, 0.8512013, 0.1763514, 1.0179800, -1.0990852, -1.3464782]),
        SRobotQ::from_array([0.2801693, 0.9165845, 0.2148244, 0.9510065, -1.0831171, -1.2620443]),
        SRobotQ::from_array([0.2177491, 0.9850859, 0.2158370, 0.8781699, -1.1183024, -1.2107089]),
        SRobotQ::from_array([0.2758391, 0.8904893, 0.1827938, 0.9697054, -1.0520904, -1.2573381]),
        SRobotQ::from_array([0.2849699, 0.7979505, 0.1046686, 0.9483294, -1.0480146, -1.3293309]),
        SRobotQ::from_array([0.1961584, 0.8614633, 0.1554078, 0.9999405, -1.1274721, -1.3711356]),
        SRobotQ::from_array([0.2676405, 0.7901247, 0.0935843, 1.0396187, -1.1730380, -1.3732298]),
        SRobotQ::from_array([0.1936202, 0.6951341, 0.0108883, 1.1101067, -1.2299142, -1.4591768]),
        SRobotQ::from_array([0.2042345, 0.7925586, -0.0770198, 1.1643602, -1.1788505, -1.3601731]),
        SRobotQ::from_array([0.1772767, 0.7440868, -0.0045473, 1.2396978, -1.2647126, -1.3096728]),
        SRobotQ::from_array([0.2500148, 0.7903014, 0.0535562, 1.1545478, -1.2553731, -1.3982640]),
        SRobotQ::from_array([0.3239738, 0.6914120, 0.1236605, 1.1396301, -1.3122255, -1.4305973]),
        SRobotQ::from_array([0.3457220, 0.7121638, 0.1943251, 1.0607872, -1.2247261, -1.4396483]),
        SRobotQ::from_array([0.2652950, 0.6738650, 0.2469884, 1.0853137, -1.3219487, -1.4402719]),
        SRobotQ::from_array([0.2848045, 0.7311544, 0.3184022, 1.1667714, -1.3546409, -1.3477301]),
        SRobotQ::from_array([0.2737518, 0.7860500, 0.4029508, 1.2577620, -1.3499184, -1.3142234]),
        SRobotQ::from_array([0.2533161, 0.8850348, 0.4109326, 1.2623707, -1.4458526, -1.3847241]),
        SRobotQ::from_array([0.2767170, 0.9426061, 0.4777802, 1.2599477, -1.4371838, -1.3086173]),
        SRobotQ::from_array([0.2284535, 0.9425596, 0.4273691, 1.2964639, -1.4957550, -1.2169469]),
        SRobotQ::from_array([0.1689166, 0.9606519, 0.3400774, 1.2576398, -1.5093432, -1.2456689]),
        SRobotQ::from_array([0.2607760, 0.8947233, 0.3993521, 1.2092380, -1.5728505, -1.1910087]),
        SRobotQ::from_array([0.2070883, 0.8881449, 0.4243983, 1.1515865, -1.6702937, -1.2591626]),
        SRobotQ::from_array([0.1682711, 0.8673760, 0.3575768, 1.1450649, -1.7561177, -1.2556389]),
        SRobotQ::from_array([0.0752571, 0.8218140, 0.4143286, 1.1751732, -1.8027944, -1.1714458]),
        SRobotQ::from_array([0.0273413, 0.9066567, 0.3907296, 1.1128650, -1.7628050, -1.1570527]),
        SRobotQ::from_array([0.0327829, 0.8962876, 0.3926617, 1.1951643, -1.6699687, -1.0671434]),
        SRobotQ::from_array([0.0685963, 0.9272223, 0.4217075, 1.2061300, -1.7079343, -0.9955891]),
        SRobotQ::from_array([0.1085894, 0.8989490, 0.4181724, 1.2784317, -1.7519532, -1.0057845]),
        SRobotQ::from_array([0.0601945, 0.8779040, 0.3456943, 1.2673952, -1.8081218, -0.9385045]),
        SRobotQ::from_array([0.0829960, 0.8381892, 0.3949460, 1.2175243, -1.7146095, -1.0313196]),
        SRobotQ::from_array([0.1520397, 0.7552742, 0.4169602, 1.1308244, -1.7895352, -1.0720215]),
        SRobotQ::from_array([0.1429673, 0.6877099, 0.4091094, 1.1654123, -1.8709186, -1.1462775]),
        SRobotQ::from_array([0.2112407, 0.6016545, 0.3173946, 1.1989725, -1.8561647, -1.0833602]),
        SRobotQ::from_array([0.1451975, 0.6494479, 0.2705453, 1.1745205, -1.8705170, -1.1463838]),
    ]
}

/// Bench multi-seg-50wp run 3/4 — 50 wp `Δ ≈ 0.1 rad` segments, 127 iter.
#[test]
fn bench_multi_seg_50wp_locally_infeasible() {
    let waypoints = vec![
        SRobotQ::from_array([0.3343222, 0.8929453, -0.0536905, 0.7182915, -1.2274445, -1.3873202]),
        SRobotQ::from_array([0.2674488, 0.9586826, -0.0377492, 0.6382554, -1.2414511, -1.3031950]),
        SRobotQ::from_array([0.3439960, 0.9346094, -0.1304580, 0.6939549, -1.2237945, -1.3037018]),
        SRobotQ::from_array([0.3770593, 0.8371893, -0.0471791, 0.7772376, -1.2808111, -1.3488145]),
        SRobotQ::from_array([0.3357331, 0.8777153, -0.0870164, 0.8393403, -1.2695598, -1.3989979]),
        SRobotQ::from_array([0.4185733, 0.9508877, -0.0808071, 0.9095306, -1.2537831, -1.3341339]),
        SRobotQ::from_array([0.4889850, 0.8706944, -0.0575931, 0.9365597, -1.3327671, -1.2880048]),
        SRobotQ::from_array([0.5089944, 0.7954211, -0.0015010, 0.9421552, -1.3384181, -1.1912302]),
        SRobotQ::from_array([0.5669821, 0.8019158, 0.0249302, 0.9983462, -1.2633414, -1.2682264]),
        SRobotQ::from_array([0.5462322, 0.8269755, 0.0114744, 1.0248954, -1.2638821, -1.1814715]),
        SRobotQ::from_array([0.4478502, 0.8240671, -0.0374530, 1.0970184, -1.2539433, -1.2531433]),
        SRobotQ::from_array([0.3921103, 0.8241266, 0.0245878, 1.1726546, -1.1683801, -1.3131421]),
        SRobotQ::from_array([0.2961473, 0.8801921, 0.0635315, 1.1099756, -1.1120828, -1.3019841]),
        SRobotQ::from_array([0.3483047, 0.8519227, 0.0110124, 1.0240691, -1.0440697, -1.2836198]),
        SRobotQ::from_array([0.2576151, 0.8472690, 0.0692917, 0.9619144, -1.0929707, -1.3585659]),
        SRobotQ::from_array([0.2756267, 0.8216234, 0.0963444, 0.9842113, -1.1360959, -1.3027613]),
        SRobotQ::from_array([0.3094045, 0.8512013, 0.1763514, 1.0179800, -1.0990852, -1.3464782]),
        SRobotQ::from_array([0.2801693, 0.9165845, 0.2148244, 0.9510065, -1.0831171, -1.2620443]),
        SRobotQ::from_array([0.2177491, 0.9850859, 0.2158370, 0.8781699, -1.1183024, -1.2107089]),
        SRobotQ::from_array([0.2758391, 0.8904893, 0.1827938, 0.9697054, -1.0520904, -1.2573381]),
        SRobotQ::from_array([0.2849699, 0.7979505, 0.1046686, 0.9483294, -1.0480146, -1.3293309]),
        SRobotQ::from_array([0.1961584, 0.8614633, 0.1554078, 0.9999405, -1.1274721, -1.3711356]),
        SRobotQ::from_array([0.2676405, 0.7901247, 0.0935843, 1.0396187, -1.1730380, -1.3732298]),
        SRobotQ::from_array([0.1936202, 0.6951341, 0.0108883, 1.1101067, -1.2299142, -1.4591768]),
        SRobotQ::from_array([0.2042345, 0.7925586, -0.0770198, 1.1643602, -1.1788505, -1.3601731]),
        SRobotQ::from_array([0.1772767, 0.7440868, -0.0045473, 1.2396978, -1.2647126, -1.3096728]),
        SRobotQ::from_array([0.2500148, 0.7903014, 0.0535562, 1.1545478, -1.2553731, -1.3982640]),
        SRobotQ::from_array([0.3239738, 0.6914120, 0.1236605, 1.1396301, -1.3122255, -1.4305973]),
        SRobotQ::from_array([0.3457220, 0.7121638, 0.1943251, 1.0607872, -1.2247261, -1.4396483]),
        SRobotQ::from_array([0.2652950, 0.6738650, 0.2469884, 1.0853137, -1.3219487, -1.4402719]),
        SRobotQ::from_array([0.2848045, 0.7311544, 0.3184022, 1.1667714, -1.3546409, -1.3477301]),
        SRobotQ::from_array([0.2737518, 0.7860500, 0.4029508, 1.2577620, -1.3499184, -1.3142234]),
        SRobotQ::from_array([0.2533161, 0.8850348, 0.4109326, 1.2623707, -1.4458526, -1.3847241]),
        SRobotQ::from_array([0.2767170, 0.9426061, 0.4777802, 1.2599477, -1.4371838, -1.3086173]),
        SRobotQ::from_array([0.2284535, 0.9425596, 0.4273691, 1.2964639, -1.4957550, -1.2169469]),
        SRobotQ::from_array([0.1689166, 0.9606519, 0.3400774, 1.2576398, -1.5093432, -1.2456689]),
        SRobotQ::from_array([0.2607760, 0.8947233, 0.3993521, 1.2092380, -1.5728505, -1.1910087]),
        SRobotQ::from_array([0.2070883, 0.8881449, 0.4243983, 1.1515865, -1.6702937, -1.2591626]),
        SRobotQ::from_array([0.1682711, 0.8673760, 0.3575768, 1.1450649, -1.7561177, -1.2556389]),
        SRobotQ::from_array([0.0752571, 0.8218140, 0.4143286, 1.1751732, -1.8027944, -1.1714458]),
        SRobotQ::from_array([0.0273413, 0.9066567, 0.3907296, 1.1128650, -1.7628050, -1.1570527]),
        SRobotQ::from_array([0.0327829, 0.8962876, 0.3926617, 1.1951643, -1.6699687, -1.0671434]),
        SRobotQ::from_array([0.0685963, 0.9272223, 0.4217075, 1.2061300, -1.7079343, -0.9955891]),
        SRobotQ::from_array([0.1085894, 0.8989490, 0.4181724, 1.2784317, -1.7519532, -1.0057845]),
        SRobotQ::from_array([0.0601945, 0.8779040, 0.3456943, 1.2673952, -1.8081218, -0.9385045]),
        SRobotQ::from_array([0.0829960, 0.8381892, 0.3949460, 1.2175243, -1.7146095, -1.0313196]),
        SRobotQ::from_array([0.1520397, 0.7552742, 0.4169602, 1.1308244, -1.7895352, -1.0720215]),
        SRobotQ::from_array([0.1429673, 0.6877099, 0.4091094, 1.1654123, -1.8709186, -1.1462775]),
        SRobotQ::from_array([0.2112407, 0.6016545, 0.3173946, 1.1989725, -1.8561647, -1.0833602]),
        SRobotQ::from_array([0.1451975, 0.6494479, 0.2705453, 1.1745205, -1.8705170, -1.1463838]),
    ];
    let ok = run_external_traj("bench_multi_seg_50wp", waypoints, external_cfg());
    assert!(ok, "still failing — check diagnostic above");
}

/// 23 waypoints, feasibility restoration failed after 4782 iter / **48 s**. This is the
/// most expensive failure in the batch — when this one passes again it'll be a big win.
#[test]
fn external_23wp_feas_restore_slow() {
    let waypoints = vec![
        SRobotQ::from_array([-1.1677809126410903, 0.6081039680542737, -1.1449571774391647, -0.9671749542519301, -1.6540773608625094, -2.503015468772706]),
        SRobotQ::from_array([-1.1543490323128383, 0.5711176059100824, -1.0894302379837058, -0.9607484689943345, -1.5492563814199907, -2.022415618274677]),
        SRobotQ::from_array([-1.1491085966905297, 0.562239382332482, -1.0766873670973025, -0.9596308772906399, -1.517799577911705, -1.910422514754583]),
        SRobotQ::from_array([-1.1431810576088948, 0.5534958206040115, -1.064360805706841, -0.9586927284159444, -1.4844229320449087, -1.8013985012876994]),
        SRobotQ::from_array([-1.1358820684382356, 0.5450208775630664, -1.0528610400613139, -0.9581464982643738, -1.4472005309242983, -1.698321686603267]),
        SRobotQ::from_array([-1.1278926655251895, 0.5366816297052199, -1.041779573413906, -0.9577878825378248, -1.408048613663473, -1.5982180979145315]),
        SRobotQ::from_array([-1.1080365314287537, 0.5208524738687081, -1.022094770671464, -0.9582313977549858, -1.3190312325103137, -1.4159403348577977]),
        SRobotQ::from_array([-1.0846782272030082, 0.5064116014895645, -1.005635595477014, -0.9599068930261864, -1.2214096534729286, -1.2580869987343946]),
        SRobotQ::from_array([-1.0723142473313911, 0.4998076912226126, -0.9985838831039099, -0.9610853279797056, -1.1715078834540646, -1.1885123779108493]),
        SRobotQ::from_array([-1.0596678113348184, 0.49352513775655327, -0.9921209220503429, -0.9624236763947869, -1.121269788825299, -1.1236339914588391]),
        SRobotQ::from_array([-1.034541398295914, 0.48207367543718044, -0.9809038850881131, -0.9653737888044509, -1.0232130958094325, -1.0082716412449007]),
        SRobotQ::from_array([-1.0132919018666655, 0.4741932991605162, -0.9742731815287109, -0.9684086733087792, -0.943397511993039, -0.9346710280275626]),
        SRobotQ::from_array([-1.0031888524970802, 0.47072007475127975, -0.971545340912701, -0.9699339437628128, -0.905917757700186, -0.9032025081676675]),
        SRobotQ::from_array([-0.9933254908562372, 0.4674645913057792, -0.969089730214475, -0.9714641998824903, -0.8695573948358455, -0.8741942449129352]),
        SRobotQ::from_array([-0.9837844448747564, 0.46449583386202015, -0.9669942137980985, -0.9730006566614277, -0.834693706450548, -0.8484653012679279]),
        SRobotQ::from_array([-0.9653439851752051, 0.45913872564152586, -0.9635269446750516, -0.9760909255527348, -0.7679562644058396, -0.8035630542060088]),
        SRobotQ::from_array([-0.9562812709716624, 0.4566112282637387, -0.9619774269783443, -0.9776441363735237, -0.7353311519271166, -0.7827491878284946]),
        SRobotQ::from_array([-0.9316778069959042, 0.45142363087019133, -0.9603925577306346, -0.9824297939071376, -0.650920804167383, -0.7459839563707484]),
        SRobotQ::from_array([-0.8797694789873962, 0.4416494126298528, -0.9587766835360718, -0.9931412680631272, -0.4766340067978659, -0.6639839493643411]),
        SRobotQ::from_array([-0.5450754044723944, 0.4279866803094008, -1.0242565496786957, -1.0806212007098888, 0.47996785314480944, 0.0611238653777355]),
        SRobotQ::from_array([-0.2410079200174257, 0.4477269276374842, -1.1286706205813533, -1.1760089281421675, 1.2699575201426159, 0.8588417464829341]),
        SRobotQ::from_array([-0.2229874928333871, 0.456881883517183, -1.1497414529892973, -1.1861853824754247, 1.3286939885556228, 1.0008264580918078]),
        SRobotQ::from_array([-0.20955299501889807, 0.5315039175710157, -1.2939024719166068, -1.2316098933657937, 1.4784915526572087, 1.9306395859837238]),
    ];
    // Cap iterations so this test can't burn 50 s of wall time per run; the producing
    // project hit ~5 k iter inside Sleipnir's restoration phase. 600 iter is plenty to
    // see whether we're closer to feasible.
    let mut cfg = external_cfg();
    cfg.solver.max_iterations = 600;
    let ok = run_external_traj("external_23wp_feas_restore_slow", waypoints, cfg);
    assert!(ok, "still failing — check diagnostic above");
}

// ----------------------------------------------------------------------------
// Second batch (2026-05-12) — three more failures from `valstad::controls::universal::traj_gen`.
// All run with `external_cfg()` (TCP v=2 a=20 j=200, default solver options w/ two-stage
// warm-start on). Captures the warning lines verbatim.
// ----------------------------------------------------------------------------

/// 3 wp, status=locally infeasible, limiting=JointVelocity, ~68 iter / 159 ms.
#[test]
fn external_3wp_jv_infeasible() {
    let waypoints = vec![
        SRobotQ::from_array([-0.5339535097453525, 0.74498675515349, -1.1050209027842832, -1.4289759983486348, 1.4386454927014827, 1.644987848470685]),
        SRobotQ::from_array([-0.5374347769801416, 0.7015257184949478, -1.0920137429213423, -0.9566055105144483, 1.1775898514577787, 1.9667187355379523]),
        SRobotQ::from_array([-0.6120284630576481, 0.6487779263898105, -1.0738133155393101, -0.3213680361203431, 0.8516805909536677, 2.389489759293551]),
    ];
    let ok = run_external_traj("external_3wp_jv_infeasible", waypoints, external_cfg());
    assert!(ok, "still failing — check diagnostic above");
}

/// 4 wp, status=locally infeasible, limiting=JointVelocity, ~371 iter / 2.2 s.
#[test]
fn external_4wp_jv_infeasible() {
    let waypoints = vec![
        SRobotQ::from_array([-0.6844456406012558, 0.27943208064805813, -1.0291380777032204, -1.0177393097241967, 0.3801949001369142, 1.0611198294268227]),
        SRobotQ::from_array([-0.7879614417204257, 0.18060696474847898, -0.9056821833396007, -0.9446100616633397, 0.17271020047616978, 0.633403057607808]),
        SRobotQ::from_array([-0.7457079161375768, 0.09388051212451275, -0.9347383979149606, 0.012300712001010052, -0.01999423969876598, -0.33654199449183686]),
        SRobotQ::from_array([-0.6140441403794081, 0.2362715059394544, -0.9127875180643983, 0.1074237519778685, -0.6194001488746936, -0.985872517054197]),
    ];
    let ok = run_external_traj("external_4wp_jv_infeasible", waypoints, external_cfg());
    assert!(ok, "still failing — check diagnostic above");
}

/// 3 wp, status=max iterations exceeded, limiting=None, ~2647 iter / 8.5 s in the field.
/// Capped to 600 iter here so the test doesn't dominate CI wall time.
#[test]
fn external_3wp_max_iter_burn() {
    let waypoints = vec![
        SRobotQ::from_array([1.5781361953476452, 0.5871822955443483, -0.2914657111179135, -0.6136122566304394, -0.30540762557051676, 0.012720444323725405]),
        SRobotQ::from_array([0.04582322042765487, 0.8358902301235366, -0.6940597972956994, 0.24628190378634335, -2.0017796241964514, -1.359778649135518]),
        SRobotQ::from_array([0.013284717572806378, 1.0293772926092564, -0.9137038190858852, 0.2351194911416346, -2.2004976102007894, -1.2531773054630062]),
    ];
    let mut cfg = external_cfg();
    cfg.solver.max_iterations = 600;
    let ok = run_external_traj("external_3wp_max_iter_burn", waypoints, cfg);
    assert!(ok, "still failing — check diagnostic above");
}

// ----------------------------------------------------------------------------
// Third batch (2026-05-12) — two more 2-waypoint p2p failures from
// `valstad::controls::universal::traj_gen`. Both are the same long-chord joint-space
// line (chord ≈ 3.564m), single segment densified to 73 samples. PCHIP on 2 colinear
// waypoints gives qpp ≈ qppp ≈ 0, so all joint a/j constraints reduce to scalar
// `|qp_j·sdd| ≤ a_max[j]` / `|qp_j·sddd| ≤ j_max[j]`; joint 0 (largest |Δq|) sets the
// tightest scalar caps, and the initial-guess sddd lands *exactly* on j0's jerk limit.
// Tiny FP perturbations between the two starts flip the IPM between "locally
// infeasible" and "feasibility restoration failed".
// ----------------------------------------------------------------------------

/// 2 wp, locally infeasible at iter 281 / 2.3 s. Chord 3.564, max_sddd in initial
/// guess = 34.022 ≈ j_max[0] / qp_0 = 22.897 / 0.673.
#[test]
fn external_2wp_long_chord_locally_infeasible() {
    let waypoints = vec![
        SRobotQ::from_array([
            -0.8204609515172723, 0.31613102569329266, -0.923207714223723,
            0.39905459054027176, -0.9012370581425307, -2.2446286935813715,
        ]),
        SRobotQ::from_array([
            1.5781361953476498, 0.5871822955443382, -0.29146571111791547,
            -0.6136122566303717, -0.30540762557050677, 0.012720444323654462,
        ]),
    ];
    let ok = run_external_traj(
        "external_2wp_long_chord_locally_infeasible",
        waypoints,
        external_cfg(),
    );
    assert!(ok, "still failing — check diagnostic above");
}

/// 2 wp, feasibility restoration failed at iter 555 / 4.3 s. Same chord/end as the
/// `_locally_infeasible` case above, start perturbed ~4 mrad in q[0,1,5] — enough to
/// flip the IPM's failure mode.
#[test]
fn external_2wp_long_chord_feas_restore_failed() {
    let waypoints = vec![
        SRobotQ::from_array([
            -0.816060612385409, 0.30785362088310586, -0.924320181316653,
            0.3980429700691374, -0.9013845276941991, -2.2484856962704693,
        ]),
        SRobotQ::from_array([
            1.5781361953476494, 0.5871822955443388, -0.29146571111791625,
            -0.6136122566303709, -0.30540762557050655, 0.012720444323652685,
        ]),
    ];
    let ok = run_external_traj(
        "external_2wp_long_chord_feas_restore_failed",
        waypoints,
        external_cfg(),
    );
    assert!(ok, "still failing — check diagnostic above");
}

// ----------------------------------------------------------------------------
// Fourth batch (2026-05-12) — bench-discovered regressions.
//
// Captured from `external_bench` run-by-seed output. The stage-1→stage-2 warm-start
// rescale (see retimer.rs::rescale_warm_start) is a net win across the bench
// distribution (50wp 75%→100%, 25wp 76→28 iter mean) but tipped two cases out of
// success: `p2p large` run 6/8 — already a pre-existing 7/8 case, captured here for
// regression — and `multi-seg 10wp` run 3/6, which previously passed and now fails
// `FeasibilityRestorationFailed`. Both go through stage 2 from a (rescaled) stage-1
// warm start that the IPM can't reconcile with TCP a/j.
// ----------------------------------------------------------------------------

/// `multi-seg 10wp` run 3/6 — was 100% before the warm-start rescale, now
/// `FeasibilityRestorationFailed` after ~320 iter. Different signature from the
/// `external_2wp_long_chord_*` "stage-2 corner" failure: this path has real curvature
/// (10 distinct waypoints, joint-space PCHIP has non-zero qpp throughout).
#[test]
fn external_bench_multi_seg_10wp_run3() {
    let waypoints = vec![
        SRobotQ::from_array([-1.1967357, 0.6513940, 0.0649984, -0.7458407, -1.0254644, 1.9914096]),
        SRobotQ::from_array([-1.4218939, 0.7337620, 0.3250841, -0.5453823, -0.9866293, 1.6930232]),
        SRobotQ::from_array([-1.5209670, 0.7668492, 0.5091862, -0.3223294, -0.8368485, 1.4805338]),
        SRobotQ::from_array([-1.3353859, 0.6944350, 0.5843410, -0.0834244, -0.9241249, 1.1985058]),
        SRobotQ::from_array([-1.5004166, 0.7278752, 0.6759865, -0.2666964, -0.7699144, 1.4176370]),
        SRobotQ::from_array([-1.3280353, 0.5349262, 0.4342674, -0.4279996, -0.9543195, 1.6290450]),
        SRobotQ::from_array([-1.2149905, 0.3504188, 0.7067139, -0.3298242, -0.9069862, 1.6441734]),
        SRobotQ::from_array([-1.2035334, 0.3318515, 1.0019210, -0.2021268, -0.6941859, 1.5207703]),
        SRobotQ::from_array([-1.3864710, 0.3202283, 1.1828311, -0.2501720, -0.8450851, 1.3589037]),
        SRobotQ::from_array([-1.5619611, 0.5108429, 1.3277226, -0.0363543, -0.6535910, 1.3245343]),
    ];
    let ok = run_external_traj(
        "external_bench_multi_seg_10wp_run3",
        waypoints,
        external_cfg(),
    );
    assert!(ok, "still failing — check diagnostic above");
}

// ----------------------------------------------------------------------------
// Two-stage warm-start experiments
//
// Hypothesis: for paths where the IPM stalls in a bad basin (e.g. `external_8wp`,
// `bench_p2p_*`), solving the TCP-disabled version first gives a feasible warm start
// that lets the TCP-enabled solve converge from a known-feasible point.
// ----------------------------------------------------------------------------

/// Run the two-stage solve on the given waypoints + cfg. Stage 1: TCP disabled (joint
/// constraints + integrator only). Stage 2: TCP enabled, seeded with stage 1's
/// `Solution`. Reports the diagnostic for each stage. Returns true iff stage 2 (the
/// "real" TCP-enabled retime) succeeds.
fn two_stage_solve(
    name: &str,
    waypoints: Vec<SRobotQ<6, f64>>,
    cfg: &Topp3Tcp6Constraints<6>,
) -> bool {
    let fk = external_chain();
    let path = SRobotPath::<6, f64>::try_new(waypoints).unwrap();

    // --- densify exactly as the retimer would so stage 1 and 2 see the same path ---
    let densified = {
        let mut p = path.densify(cfg.densification.max_segment_step.unwrap_or(0.05));
        if p.len() < cfg.densification.min_samples {
            let n = cfg.densification.min_samples.max(2);
            let mut wps = Vec::with_capacity(n);
            for i in 0..n {
                let t = i as f64 / (n - 1) as f64;
                wps.push(p.sample(t).unwrap_or(*p.first()));
            }
            p = SRobotPath::try_new(wps).unwrap();
        }
        if p.len() > cfg.densification.max_samples {
            let n = cfg.densification.max_samples.max(2);
            let mut wps = Vec::with_capacity(n);
            for i in 0..n {
                let t = i as f64 / (n - 1) as f64;
                wps.push(p.sample(t).unwrap_or(*p.first()));
            }
            p = SRobotPath::try_new(wps).unwrap();
        }
        p
    };

    // --- stage 1: TCP disabled ---
    let deriv_no_tcp = PathDerivatives::<6>::new_without_tcp(&densified).unwrap();
    let start_no_tcp = boundary::project::<6>(
        &cfg.boundary.v_start,
        &cfg.boundary.a_start,
        &deriv_no_tcp.qp[0],
        &deriv_no_tcp.qpp[0],
    );
    let end_no_tcp = boundary::project::<6>(
        &cfg.boundary.v_end,
        &cfg.boundary.a_end,
        &deriv_no_tcp.qp[deriv_no_tcp.num_waypoints() - 1],
        &deriv_no_tcp.qpp[deriv_no_tcp.num_waypoints() - 1],
    );

    let mut cfg_no_tcp = cfg.clone();
    cfg_no_tcp.tcp = None;

    let stage1 = build_and_solve(&deriv_no_tcp, &cfg_no_tcp, start_no_tcp, end_no_tcp).unwrap();
    eprintln!(
        "[{}] stage 1 (no TCP): status={:?} iter={} solve={:.3}s",
        name,
        stage1.status,
        stage1.iterations,
        stage1.solve_time.as_secs_f64()
    );
    if !matches!(stage1.status, SolveStatus::Success) {
        eprintln!("  stage 1 failed; not running stage 2");
        return false;
    }

    // --- stage 2: TCP enabled, warm-started from stage 1 ---
    let deriv = PathDerivatives::<6>::new(&densified, &fk).unwrap();
    let start = boundary::project::<6>(
        &cfg.boundary.v_start,
        &cfg.boundary.a_start,
        &deriv.qp[0],
        &deriv.qpp[0],
    );
    let end = boundary::project::<6>(
        &cfg.boundary.v_end,
        &cfg.boundary.a_end,
        &deriv.qp[deriv.num_waypoints() - 1],
        &deriv.qpp[deriv.num_waypoints() - 1],
    );
    let stage2 = build_and_solve_warm(&deriv, cfg, start, end, &stage1).unwrap();
    eprintln!(
        "[{}] stage 2 (TCP, warm): status={:?} iter={} solve={:.3}s",
        name,
        stage2.status,
        stage2.iterations,
        stage2.solve_time.as_secs_f64()
    );
    matches!(stage2.status, SolveStatus::Success)
}

/// Smoke test the two-stage helper itself with a healthy path.
#[test]
fn two_stage_smoke() {
    let waypoints = vec![
        SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
        SRobotQ::from_array([0.3, -0.7, 0.9, 0.2, 0.1, 0.3]),
    ];
    let ok = two_stage_solve("two_stage_smoke", waypoints, &external_cfg());
    assert!(ok, "two-stage smoke failed");
}

/// Try `external_8wp` (the stuck-basin failure that consumes any iter budget) with the
/// two-stage warm-start strategy.
#[test]
fn two_stage_8wp() {
    let waypoints = vec![
        SRobotQ::from_array([-0.6364632544213349, 0.3802690502211359, -0.9972162375620871, -1.1558150171008872, 0.3874953228516746, 1.1660525739447671]),
        SRobotQ::from_array([-0.721846280710852, 0.37576783004126607, -0.9307956849654027, -0.7378766433380558, 0.3050033760616249, 1.1802386380717917]),
        SRobotQ::from_array([-0.7504928867152842, 0.3708009486130261, -0.9006694236872432, -0.5975635473851985, 0.27467197128185794, 1.1677764895126956]),
        SRobotQ::from_array([-0.7502668817422785, 0.3626875882640009, -0.8878925771004293, -0.5771302319628522, 0.2635020161312982, 1.118628367931161]),
        SRobotQ::from_array([-0.6057486690247245, 0.17619503374059908, -0.864866663660868, 0.39110067478216165, -0.16813003301110035, -0.8783368611747007]),
        SRobotQ::from_array([-0.5815544282609763, 0.15464996057632047, -0.8760812096089616, 0.5296678139464301, -0.22625197515831497, -1.1552106572603036]),
        SRobotQ::from_array([-0.5572599641132697, 0.13311599806229463, -0.8873891827746738, 0.668517912042139, -0.28417834246355345, -1.4324307081889254]),
        SRobotQ::from_array([-0.4634180866210337, 0.25899651320682565, -0.9699935892427178, 1.158330849209562, -0.44912260485450173, -1.582155970300871]),
    ];
    let ok = two_stage_solve("two_stage_8wp", waypoints, &external_cfg());
    // Don't assert success — this is exploratory. The eprintln tells the story.
    let _ = ok;
}

/// `bench_multi_seg_50wp` — uses the same captured 50wp path from the bench.
#[test]
fn two_stage_50wp() {
    let waypoints = bench_multi_seg_50wp_waypoints();
    let ok = two_stage_solve("two_stage_50wp", waypoints, &external_cfg());
    let _ = ok;
}

/// `bench_p2p_small` — the captured tight-jerk 2wp path.
#[test]
fn two_stage_p2p_small() {
    let waypoints = vec![
        SRobotQ::from_array([-1.1703483, -0.3139854, 0.1235649, -1.0546926, 0.1807552, -0.9767784]),
        SRobotQ::from_array([-1.3687564, -0.4719839, 0.3012058, -1.1227008, 0.1640270, -0.9921301]),
    ];
    let ok = two_stage_solve("two_stage_p2p_small", waypoints, &external_cfg());
    let _ = ok;
}

/// Diagnostic: loops two-stage solves through a small set of *different* paths
/// (rotated through a 4-path cycle for ~80 invocations) to see whether varying the
/// problem shape between calls is what triggers the SIGSEGV. If this crashes and
/// `two_stage_repeat_one_path_100x` doesn't, the bug needs a shape-change between
/// consecutive arenas.
#[test]
fn two_stage_repeat_varied_paths() {
    let paths = vec![
        // 2wp
        vec![
            SRobotQ::from_array([0.0, -1.0, 1.2, 0.0, 0.0, 0.0]),
            SRobotQ::from_array([0.3, -0.7, 0.9, 0.2, 0.1, 0.3]),
        ],
        // 8wp captured
        vec![
            SRobotQ::from_array([-0.6364632544213349, 0.3802690502211359, -0.9972162375620871, -1.1558150171008872, 0.3874953228516746, 1.1660525739447671]),
            SRobotQ::from_array([-0.721846280710852, 0.37576783004126607, -0.9307956849654027, -0.7378766433380558, 0.3050033760616249, 1.1802386380717917]),
            SRobotQ::from_array([-0.7504928867152842, 0.3708009486130261, -0.9006694236872432, -0.5975635473851985, 0.27467197128185794, 1.1677764895126956]),
            SRobotQ::from_array([-0.7502668817422785, 0.3626875882640009, -0.8878925771004293, -0.5771302319628522, 0.2635020161312982, 1.118628367931161]),
            SRobotQ::from_array([-0.6057486690247245, 0.17619503374059908, -0.864866663660868, 0.39110067478216165, -0.16813003301110035, -0.8783368611747007]),
            SRobotQ::from_array([-0.5815544282609763, 0.15464996057632047, -0.8760812096089616, 0.5296678139464301, -0.22625197515831497, -1.1552106572603036]),
            SRobotQ::from_array([-0.5572599641132697, 0.13311599806229463, -0.8873891827746738, 0.668517912042139, -0.28417834246355345, -1.4324307081889254]),
            SRobotQ::from_array([-0.4634180866210337, 0.25899651320682565, -0.9699935892427178, 1.158330849209562, -0.44912260485450173, -1.582155970300871]),
        ],
        // 3wp
        vec![
            SRobotQ::from_array([0.1, -1.1, 1.0, 0.0, 0.0, 0.0]),
            SRobotQ::from_array([0.2, -0.9, 1.0, 0.1, 0.0, 0.1]),
            SRobotQ::from_array([0.3, -0.7, 0.9, 0.2, 0.1, 0.2]),
        ],
        // 50wp captured (uses helper)
        bench_multi_seg_50wp_waypoints(),
    ];
    let cfg = external_cfg();
    let n_iter = 80;
    let mut successes = 0_usize;
    for i in 0..n_iter {
        let wps = paths[i % paths.len()].clone();
        let ok = two_stage_solve(&format!("varied #{i} idx={}", i % paths.len()), wps, &cfg);
        if ok {
            successes += 1;
        }
    }
    eprintln!("two_stage_repeat_varied_paths: {successes}/{n_iter} succeeded");
}

/// Diagnostic: loops the *same* 8wp trajectory through two-stage 100 times in a tight
/// loop. If this crashes, the bug isn't dataset-dependent — it's purely from rapid
/// arena create/drop pairs. If it doesn't crash, the bug needs a particular sequence
/// of paths to manifest. (Run with `--release` since the crash only reproduces there.)
#[test]
fn two_stage_repeat_one_path_100x() {
    let waypoints = vec![
        SRobotQ::from_array([-0.6364632544213349, 0.3802690502211359, -0.9972162375620871, -1.1558150171008872, 0.3874953228516746, 1.1660525739447671]),
        SRobotQ::from_array([-0.721846280710852, 0.37576783004126607, -0.9307956849654027, -0.7378766433380558, 0.3050033760616249, 1.1802386380717917]),
        SRobotQ::from_array([-0.7504928867152842, 0.3708009486130261, -0.9006694236872432, -0.5975635473851985, 0.27467197128185794, 1.1677764895126956]),
        SRobotQ::from_array([-0.7502668817422785, 0.3626875882640009, -0.8878925771004293, -0.5771302319628522, 0.2635020161312982, 1.118628367931161]),
        SRobotQ::from_array([-0.6057486690247245, 0.17619503374059908, -0.864866663660868, 0.39110067478216165, -0.16813003301110035, -0.8783368611747007]),
        SRobotQ::from_array([-0.5815544282609763, 0.15464996057632047, -0.8760812096089616, 0.5296678139464301, -0.22625197515831497, -1.1552106572603036]),
        SRobotQ::from_array([-0.5572599641132697, 0.13311599806229463, -0.8873891827746738, 0.668517912042139, -0.28417834246355345, -1.4324307081889254]),
        SRobotQ::from_array([-0.4634180866210337, 0.25899651320682565, -0.9699935892427178, 1.158330849209562, -0.44912260485450173, -1.582155970300871]),
    ];
    let cfg = external_cfg();
    let mut successes = 0_usize;
    let n_iter = 100;
    for i in 0..n_iter {
        let ok = two_stage_solve(&format!("repeat #{i}"), waypoints.clone(), &cfg);
        if ok {
            successes += 1;
        }
    }
    eprintln!("two_stage_repeat_one_path_100x: {successes}/{n_iter} succeeded");
    assert!(successes >= n_iter - 1, "expected almost all to succeed");
}
