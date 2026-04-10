use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{Expr, LitStr, Token};

struct CricketConfig {
    name: String,
    urdf: String,
    srdf: String,
    end_effector: String,
    forced_end_effector_collision: Vec<String>,
    ignored_environment_collision: Vec<String>,
    sphere_padding: f32,
    sphere_only: bool,
}

#[derive(Debug, Clone)]
enum ShapeData {
    Sphere {
        x: f32,
        y: f32,
        z: f32,
        radius: f32,
    },
    Cylinder {
        x: f32,
        y: f32,
        z: f32,
        rpy: [f32; 3],
        radius: f32,
        length: f32,
    },
    Box {
        x: f32,
        y: f32,
        z: f32,
        rpy: [f32; 3],
        sx: f32,
        sy: f32,
        sz: f32,
    },
}

impl ShapeData {
    fn has_rotation(&self) -> bool {
        match self {
            ShapeData::Sphere { .. } => false,
            ShapeData::Cylinder { rpy, .. } | ShapeData::Box { rpy, .. } => {
                rpy[0].abs() > 1e-8 || rpy[1].abs() > 1e-8 || rpy[2].abs() > 1e-8
            }
        }
    }
}

#[derive(Debug)]
struct LinkData {
    name: String,
    shapes: Vec<ShapeData>,
}

#[derive(Debug)]
struct JointData {
    name: String,
    parent: String,
    child: String,
    origin_xyz: [f64; 3],
    origin_rpy: [f64; 3],
    axis: [f64; 3],
    lower: f64,
    upper: f64,
    velocity: f64,
}

struct MacroInput {
    config: CricketConfig,
}

impl MacroInput {
    fn parse_string_list(input: ParseStream) -> syn::Result<Vec<String>> {
        let content;
        syn::bracketed!(content in input);
        let mut items = Vec::new();
        while !content.is_empty() {
            let lit: LitStr = content.parse()?;
            items.push(lit.value());
            if !content.is_empty() {
                content.parse::<Token![,]>()?;
            }
        }
        Ok(items)
    }
}

impl Parse for MacroInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut urdf = None;
        let mut srdf = None;
        let mut end_effector = None;
        let mut forced_end_effector_collision = Vec::new();
        let mut ignored_environment_collision = Vec::new();
        let mut sphere_padding = 0.0f32;
        let mut sphere_only = false;

        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "name" => {
                    let val: LitStr = input.parse()?;
                    name = Some(val.value());
                }
                "urdf" => {
                    let val: LitStr = input.parse()?;
                    urdf = Some(val.value());
                }
                "srdf" => {
                    let val: LitStr = input.parse()?;
                    srdf = Some(val.value());
                }
                "end_effector" => {
                    let val: LitStr = input.parse()?;
                    end_effector = Some(val.value());
                }
                "forced_end_effector_collision" => {
                    forced_end_effector_collision = Self::parse_string_list(input)?;
                }
                "ignored_environment_collision" => {
                    ignored_environment_collision = Self::parse_string_list(input)?;
                }
                "sphere_padding" => {
                    let val: syn::LitFloat = input.parse()?;
                    sphere_padding = val.base10_parse()?;
                }
                "sphere_only" => {
                    let val: syn::LitBool = input.parse()?;
                    sphere_only = val.value;
                }
                other => {
                    return Err(syn::Error::new(key.span(), format!("unknown argument `{}`", other)));
                }
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        let config = CricketConfig {
            name: name.ok_or_else(|| input.error("missing required argument `name`"))?,
            urdf: urdf.ok_or_else(|| input.error("missing required argument `urdf`"))?,
            srdf: srdf.ok_or_else(|| input.error("missing required argument `srdf`"))?,
            end_effector: end_effector
                .ok_or_else(|| input.error("missing required argument `end_effector`"))?,
            forced_end_effector_collision,
            ignored_environment_collision,
            sphere_padding,
            sphere_only,
        };

        Ok(MacroInput { config })
    }
}

fn parse_xyz(s: &str) -> [f64; 3] {
    let parts: Vec<f64> = s.split_whitespace().map(|p| p.parse().unwrap()).collect();
    [parts[0], parts[1], parts[2]]
}

fn parse_origin(col: &roxmltree::Node) -> ([f32; 3], [f32; 3]) {
    let origin = col.children().find(|n| n.tag_name().name() == "origin");
    if let Some(origin) = origin {
        let xyz = parse_xyz(origin.attribute("xyz").unwrap_or("0 0 0"));
        let rpy = parse_xyz(origin.attribute("rpy").unwrap_or("0 0 0"));
        (
            [xyz[0] as f32, xyz[1] as f32, xyz[2] as f32],
            [rpy[0] as f32, rpy[1] as f32, rpy[2] as f32],
        )
    } else {
        ([0.0; 3], [0.0; 3])
    }
}

fn parse_urdf(xml: &str) -> (Vec<LinkData>, Vec<JointData>) {
    let doc = roxmltree::Document::parse(xml).expect("failed to parse URDF");
    let root = doc.root_element();

    let mut links = Vec::new();
    let mut joints = Vec::new();

    for node in root.children() {
        if node.tag_name().name() == "link" {
            let name = node.attribute("name").unwrap().to_string();
            let mut shapes = Vec::new();
            for col in node
                .children()
                .filter(|n| n.tag_name().name() == "collision")
            {
                let geom = col
                    .children()
                    .find(|n| n.tag_name().name() == "geometry")
                    .unwrap();
                let ([x, y, z], rpy) = parse_origin(&col);

                if let Some(sphere) = geom.children().find(|n| n.tag_name().name() == "sphere") {
                    let radius: f32 = sphere.attribute("radius").unwrap().parse().unwrap();
                    shapes.push(ShapeData::Sphere { x, y, z, radius });
                } else if let Some(cyl) =
                    geom.children().find(|n| n.tag_name().name() == "cylinder")
                {
                    let radius: f32 = cyl.attribute("radius").unwrap().parse().unwrap();
                    let length: f32 = cyl.attribute("length").unwrap().parse().unwrap();
                    shapes.push(ShapeData::Cylinder {
                        x,
                        y,
                        z,
                        rpy,
                        radius,
                        length,
                    });
                } else if let Some(bx) = geom.children().find(|n| n.tag_name().name() == "box") {
                    let size = parse_xyz(bx.attribute("size").unwrap());
                    shapes.push(ShapeData::Box {
                        x,
                        y,
                        z,
                        rpy,
                        sx: size[0] as f32,
                        sy: size[1] as f32,
                        sz: size[2] as f32,
                    });
                }
            }
            links.push(LinkData { name, shapes });
        } else if node.tag_name().name() == "joint" {
            let joint_type = node.attribute("type").unwrap_or("fixed");
            if joint_type != "revolute" && joint_type != "continuous" && joint_type != "prismatic" {
                continue;
            }
            let name = node.attribute("name").unwrap().to_string();
            let parent = node
                .children()
                .find(|n| n.tag_name().name() == "parent")
                .unwrap()
                .attribute("link")
                .unwrap()
                .to_string();
            let child = node
                .children()
                .find(|n| n.tag_name().name() == "child")
                .unwrap()
                .attribute("link")
                .unwrap()
                .to_string();
            let origin = node.children().find(|n| n.tag_name().name() == "origin");
            let (origin_xyz, origin_rpy) = if let Some(o) = origin {
                (
                    parse_xyz(o.attribute("xyz").unwrap_or("0 0 0")),
                    parse_xyz(o.attribute("rpy").unwrap_or("0 0 0")),
                )
            } else {
                ([0.0; 3], [0.0; 3])
            };
            let axis = node
                .children()
                .find(|n| n.tag_name().name() == "axis")
                .map(|a| parse_xyz(a.attribute("xyz").unwrap_or("0 0 1")))
                .unwrap_or([0.0, 0.0, 1.0]);
            let limit = node.children().find(|n| n.tag_name().name() == "limit");
            let (lower, upper, velocity) = if let Some(l) = limit {
                (
                    l.attribute("lower").unwrap_or("0").parse().unwrap(),
                    l.attribute("upper").unwrap_or("0").parse().unwrap(),
                    l.attribute("velocity").unwrap_or("0").parse().unwrap(),
                )
            } else {
                (0.0, 0.0, 0.0)
            };
            joints.push(JointData {
                name,
                parent,
                child,
                origin_xyz,
                origin_rpy,
                axis,
                lower,
                upper,
                velocity,
            });
        }
    }

    (links, joints)
}

fn parse_srdf(xml: &str) -> HashSet<(String, String)> {
    let doc = roxmltree::Document::parse(xml).expect("failed to parse SRDF");
    let root = doc.root_element();
    let mut disabled = HashSet::new();

    for node in root.children() {
        if node.tag_name().name() == "disable_collisions" {
            let l1 = node.attribute("link1").unwrap().to_string();
            let l2 = node.attribute("link2").unwrap().to_string();
            let pair = if l1 < l2 { (l1, l2) } else { (l2, l1) };
            disabled.insert(pair);
        }
    }

    disabled
}

fn compute_dh_from_joints(joints: &[JointData]) -> Vec<(f64, f64, f64, f64)> {
    let mut dh_params = Vec::new();

    for joint in joints {
        let [ox, oy, oz] = joint.origin_xyz;
        let axis = joint.axis;

        let a;
        let d;
        let alpha;
        let theta_offset;

        if axis[2].abs() > 0.5 {
            a = (ox * ox + oy * oy).sqrt();
            d = oz;
            alpha = 0.0;
            theta_offset = if a.abs() > 1e-10 { oy.atan2(ox) } else { 0.0 };
        } else if axis[1].abs() > 0.5 {
            if axis[1] > 0.0 {
                a = (ox * ox + oz * oz).sqrt();
                d = oy;
                alpha = std::f64::consts::FRAC_PI_2;
                theta_offset = 0.0;
            } else {
                a = (ox * ox + oz * oz).sqrt();
                d = -oy;
                alpha = -std::f64::consts::FRAC_PI_2;
                theta_offset = 0.0;
            }
        } else if axis[0].abs() > 0.5 {
            if axis[0] > 0.0 {
                a = ox;
                d = (oy * oy + oz * oz).sqrt();
                alpha = 0.0;
                theta_offset = -std::f64::consts::FRAC_PI_2;
            } else {
                a = ox;
                d = -(oy * oy + oz * oz).sqrt();
                alpha = 0.0;
                theta_offset = std::f64::consts::FRAC_PI_2;
            }
        } else {
            a = 0.0;
            d = 0.0;
            alpha = 0.0;
            theta_offset = 0.0;
        }

        dh_params.push((a, alpha, d, theta_offset));
    }

    dh_params
}

fn build_filter(
    link_idx: usize,
    n_joints: usize,
    joint_child_names: &[String],
    ee_name: &str,
    base_name: &str,
    disabled_pairs: &HashSet<(String, String)>,
    forced_ee: &HashSet<String>,
    ignored_env: &HashSet<String>,
) -> (Vec<bool>, bool, bool, bool) {
    let my_name = if link_idx < n_joints {
        &joint_child_names[link_idx]
    } else if link_idx == n_joints {
        ee_name
    } else {
        base_name
    };

    let mut links = vec![false; n_joints];
    for other_idx in 0..n_joints {
        if other_idx == link_idx {
            continue;
        }
        let other_name = &joint_child_names[other_idx];
        let pair = if my_name < other_name.as_str() {
            (my_name.to_string(), other_name.clone())
        } else {
            (other_name.clone(), my_name.to_string())
        };
        links[other_idx] = !disabled_pairs.contains(&pair);
    }

    let ee = if link_idx < n_joints {
        let pair = if my_name < ee_name {
            (my_name.to_string(), ee_name.to_string())
        } else {
            (ee_name.to_string(), my_name.to_string())
        };
        let disabled = disabled_pairs.contains(&pair);
        let forced = forced_ee.contains(my_name);
        !disabled || forced
    } else {
        false
    };

    let base = if link_idx != n_joints + 1 && my_name != base_name {
        let pair = if my_name < base_name {
            (my_name.to_string(), base_name.to_string())
        } else {
            (base_name.to_string(), my_name.to_string())
        };
        !disabled_pairs.contains(&pair)
    } else {
        false
    };

    let obstacles = !ignored_env.contains(my_name);

    (links, ee, obstacles, base)
}

/// Rotation matrix columns from URDF roll-pitch-yaw (applied as Rz(yaw) * Ry(pitch) * Rx(roll)).
/// Returns (col0, col1, col2) as [f32; 3] each.
fn rpy_to_axes(rpy: [f32; 3]) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let (sr, cr) = (rpy[0] as f64).sin_cos();
    let (sp, cp) = (rpy[1] as f64).sin_cos();
    let (sy, cy) = (rpy[2] as f64).sin_cos();

    let col0 = [(cy * cp) as f32, (sy * cp) as f32, (-sp) as f32];
    let col1 = [
        (cy * sp * sr - sy * cr) as f32,
        (sy * sp * sr + cy * cr) as f32,
        (cp * sr) as f32,
    ];
    let col2 = [
        (cy * sp * cr + sy * sr) as f32,
        (sy * sp * cr - cy * sr) as f32,
        (cp * cr) as f32,
    ];
    (col0, col1, col2)
}

/// Generate a const token for a shape. Spheres are always const. Cylinders and
/// cuboids without rotation are const. Returns None if the shape needs runtime
/// construction (has rotation).
fn shape_const_token(shape: &ShapeData, padding: f32) -> Option<(TokenStream2, &'static str)> {
    match shape {
        ShapeData::Sphere { x, y, z, radius } => {
            let r = radius + padding;
            let tok = quote! {
                wreck::Sphere::new(glam::Vec3::new(#x, #y, #z), #r)
            };
            Some((tok, "sphere"))
        }
        ShapeData::Cylinder {
            x,
            y,
            z,
            rpy,
            radius,
            length,
        } if !shape.has_rotation() => {
            let half = length / 2.0;
            let p1z = z - half;
            let p2z = z + half;
            let tok = quote! {
                wreck::Cylinder::new(
                    glam::Vec3::new(#x, #y, #p1z),
                    glam::Vec3::new(#x, #y, #p2z),
                    #radius,
                )
            };
            Some((tok, "cylinder"))
        }
        ShapeData::Box {
            x,
            y,
            z,
            rpy,
            sx,
            sy,
            sz,
        } if !shape.has_rotation() => {
            let hx = sx / 2.0;
            let hy = sy / 2.0;
            let hz = sz / 2.0;
            let tok = quote! {
                wreck::Cuboid::from_aabb(
                    glam::Vec3::new(#x - #hx, #y - #hy, #z - #hz),
                    glam::Vec3::new(#x + #hx, #y + #hy, #z + #hz),
                )
            };
            Some((tok, "cuboid"))
        }
        _ => None,
    }
}

/// Generate a runtime add statement for a shape that has rotation.
fn shape_runtime_token(shape: &ShapeData, padding: f32) -> TokenStream2 {
    match shape {
        ShapeData::Sphere { x, y, z, radius } => {
            let r = radius + padding;
            quote! {
                collider.add(wreck::Sphere::new(glam::Vec3::new(#x, #y, #z), #r));
            }
        }
        ShapeData::Cylinder {
            x,
            y,
            z,
            rpy,
            radius,
            length,
        } => {
            let half = length / 2.0;
            let (.., c2) = rpy_to_axes(*rpy);
            // Cylinder along local Z: endpoints at origin ± half * col2, then translate
            let p1x = x + c2[0] * (-half);
            let p1y = y + c2[1] * (-half);
            let p1z = z + c2[2] * (-half);
            let p2x = x + c2[0] * half;
            let p2y = y + c2[1] * half;
            let p2z = z + c2[2] * half;
            quote! {
                collider.add(wreck::Cylinder::new(
                    glam::Vec3::new(#p1x, #p1y, #p1z),
                    glam::Vec3::new(#p2x, #p2y, #p2z),
                    #radius,
                ));
            }
        }
        ShapeData::Box {
            x,
            y,
            z,
            rpy,
            sx,
            sy,
            sz,
        } => {
            let hx = sx / 2.0;
            let hy = sy / 2.0;
            let hz = sz / 2.0;
            let (c0, c1, c2) = rpy_to_axes(*rpy);
            let c0x = c0[0];
            let c0y = c0[1];
            let c0z = c0[2];
            let c1x = c1[0];
            let c1y = c1[1];
            let c1z = c1[2];
            let c2x = c2[0];
            let c2y = c2[1];
            let c2z = c2[2];
            quote! {
                collider.add(wreck::Cuboid::new(
                    glam::Vec3::new(#x, #y, #z),
                    [
                        glam::Vec3::new(#c0x, #c0y, #c0z),
                        glam::Vec3::new(#c1x, #c1y, #c1z),
                        glam::Vec3::new(#c2x, #c2y, #c2z),
                    ],
                    [#hx, #hy, #hz],
                ));
            }
        }
    }
}

/// Generate const arrays and runtime add statements for a link's shapes.
/// Returns (const_definitions, collider_build_tokens).
fn gen_link_shapes(
    prefix: &str,
    shapes: &[ShapeData],
    padding: f32,
) -> (Vec<TokenStream2>, Vec<TokenStream2>) {
    let mut const_spheres: Vec<TokenStream2> = Vec::new();
    let mut const_cylinders: Vec<TokenStream2> = Vec::new();
    let mut const_cuboids: Vec<TokenStream2> = Vec::new();
    let mut runtime_adds: Vec<TokenStream2> = Vec::new();

    for shape in shapes {
        if let Some((tok, kind)) = shape_const_token(shape, padding) {
            match kind {
                "sphere" => const_spheres.push(tok),
                "cylinder" => const_cylinders.push(tok),
                "cuboid" => const_cuboids.push(tok),
                _ => unreachable!(),
            }
        } else {
            runtime_adds.push(shape_runtime_token(shape, padding));
        }
    }

    let mut const_defs = Vec::new();
    let mut build_stmts = Vec::new();

    if !const_spheres.is_empty() {
        let name = format_ident!("{}_SPHERES", prefix);
        let n = proc_macro2::Literal::usize_unsuffixed(const_spheres.len());
        const_defs.push(quote! {
            const #name: [wreck::Sphere; #n] = [#(#const_spheres),*];
        });
        build_stmts.push(quote! { collider.add_slice(&#name); });
    }

    if !const_cylinders.is_empty() {
        let name = format_ident!("{}_CYLINDERS", prefix);
        let n = proc_macro2::Literal::usize_unsuffixed(const_cylinders.len());
        const_defs.push(quote! {
            const #name: [wreck::Cylinder; #n] = [#(#const_cylinders),*];
        });
        build_stmts.push(quote! { collider.add_slice(&#name); });
    }

    if !const_cuboids.is_empty() {
        let name = format_ident!("{}_CUBOIDS", prefix);
        let n = proc_macro2::Literal::usize_unsuffixed(const_cuboids.len());
        const_defs.push(quote! {
            const #name: [wreck::Cuboid; #n] = [#(#const_cuboids),*];
        });
        build_stmts.push(quote! { collider.add_slice(&#name); });
    }

    build_stmts.extend(runtime_adds);

    (const_defs, build_stmts)
}

#[proc_macro]
pub fn cricket(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as MacroInput);
    let config = input.config;

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let urdf_path = PathBuf::from(&manifest_dir).join(&config.urdf);
    let srdf_path = PathBuf::from(&manifest_dir).join(&config.srdf);

    let urdf_content = std::fs::read_to_string(&urdf_path)
        .expect(&format!("failed to read {}", urdf_path.display()));
    let srdf_content = std::fs::read_to_string(&srdf_path)
        .expect(&format!("failed to read {}", srdf_path.display()));

    let (links, joints) = parse_urdf(&urdf_content);
    let disabled_pairs = parse_srdf(&srdf_content);

    let n_joints = joints.len();
    let n_lit = proc_macro2::Literal::usize_unsuffixed(n_joints);

    let mod_name = format_ident!("{}", config.name.to_lowercase());

    let link_map: HashMap<String, usize> = links
        .iter()
        .enumerate()
        .map(|(i, l)| (l.name.clone(), i))
        .collect();

    let joint_child_names: Vec<String> = joints.iter().map(|j| j.child.clone()).collect();

    let ee_link_name = &config.end_effector;
    let ee_link_idx = link_map.get(ee_link_name);

    let forced_ee: HashSet<String> = config.forced_end_effector_collision.into_iter().collect();
    let ignored_env: HashSet<String> = config.ignored_environment_collision.into_iter().collect();
    let sphere_padding = config.sphere_padding;

    let dh_params = compute_dh_from_joints(&joints);
    let dh_joint_tokens: Vec<TokenStream2> = dh_params
        .iter()
        .map(|(a, alpha, d, theta_offset)| {
            let a = *a as f32;
            let alpha = *alpha as f32;
            let d = *d as f32;
            let theta_offset = *theta_offset as f32;
            quote! {
                deke_types::DHJoint { a: #a, alpha: #alpha, d: #d, theta_offset: #theta_offset }
            }
        })
        .collect();

    let urdf_joint_tokens: Vec<TokenStream2> = joints
        .iter()
        .map(|j| {
            let ox = j.origin_xyz[0];
            let oy = j.origin_xyz[1];
            let oz = j.origin_xyz[2];
            let roll = j.origin_rpy[0];
            let pitch = j.origin_rpy[1];
            let yaw = j.origin_rpy[2];
            let ax = j.axis[0];
            let ay = j.axis[1];
            let az = j.axis[2];
            quote! {
                deke_types::URDFJoint {
                    origin_xyz: [#ox, #oy, #oz],
                    origin_rpy: [#roll, #pitch, #yaw],
                    axis: [#ax, #ay, #az],
                }
            }
        })
        .collect();

    let lower_vals: Vec<f32> = joints.iter().map(|j| j.lower as f32).collect();
    let upper_vals: Vec<f32> = joints.iter().map(|j| j.upper as f32).collect();
    let velocity_vals: Vec<f32> = joints.iter().map(|j| j.velocity as f32).collect();

    let base_link_name = &joints[0].parent;
    let ee_is_joint_child = joint_child_names.contains(&ee_link_name.to_string());

    let mut all_const_defs = Vec::new();
    let mut link_collider_builds = Vec::new();

    for (joint_idx, joint) in joints.iter().enumerate() {
        let child_link_idx = link_map[&joint.child];
        let link = &links[child_link_idx];
        let prefix = format!("LINK_{}", joint_idx);

        let (const_defs, build_stmts) = gen_link_shapes(&prefix, &link.shapes, sphere_padding);
        all_const_defs.extend(const_defs);

        let (filter_links, filter_ee, filter_obstacles, filter_base) = build_filter(
            joint_idx,
            n_joints,
            &joint_child_names,
            ee_link_name,
            base_link_name,
            &disabled_pairs,
            &forced_ee,
            &ignored_env,
        );

        let filter_link_bools: Vec<TokenStream2> = filter_links
            .iter()
            .map(|b| {
                if *b {
                    quote! { true }
                } else {
                    quote! { false }
                }
            })
            .collect();

        let has_shapes = !link.shapes.is_empty();
        let collider_expr = if has_shapes {
            quote! {
                {
                    let mut collider = wreck::Collider::default();
                    #(#build_stmts)*
                    collider.refine_bounding();
                    Some(collider)
                }
            }
        } else {
            quote! { None }
        };

        link_collider_builds.push(quote! {
            {
                deke_wreck::CollisionBody::new(
                    #collider_expr,
                    deke_wreck::CollisionFilter {
                        links: [#(#filter_link_bools),*],
                        ee: #filter_ee,
                        base: #filter_base,
                        obstacles: #filter_obstacles,
                    },
                    Vec::new(),
                    #joint_idx as i16,
                )
            }
        });
    }

    // End effector — if the ee link is already a joint child, its shapes are
    // already in the corresponding link body, so generate an empty ee body.
    let ee_collider_build;

    if !ee_is_joint_child {
        if let Some(&ee_idx) = ee_link_idx {
            let ee_link = &links[ee_idx];
            let (const_defs, build_stmts) = gen_link_shapes("EE", &ee_link.shapes, sphere_padding);
            all_const_defs.extend(const_defs);

            let (ee_filter_links, _, ee_filter_obstacles, ee_filter_base) = build_filter(
                n_joints,
                n_joints,
                &joint_child_names,
                ee_link_name,
                base_link_name,
                &disabled_pairs,
                &forced_ee,
                &ignored_env,
            );

            let ee_filter_bools: Vec<TokenStream2> = ee_filter_links
                .iter()
                .map(|b| {
                    if *b {
                        quote! { true }
                    } else {
                        quote! { false }
                    }
                })
                .collect();

            let has_shapes = !ee_link.shapes.is_empty();
            let ee_expr = if has_shapes {
                quote! {
                    {
                        let mut collider = wreck::Collider::default();
                        #(#build_stmts)*
                        Some(collider)
                    }
                }
            } else {
                quote! { None }
            };

            ee_collider_build = quote! {
                deke_wreck::CollisionBody::new(
                    #ee_expr,
                    deke_wreck::CollisionFilter {
                        links: [#(#ee_filter_bools),*],
                        ee: false,
                        base: #ee_filter_base,
                        obstacles: #ee_filter_obstacles,
                    },
                    Vec::new(),
                    #n_joints as i16 + 1,
                )
            };
        } else {
            ee_collider_build = quote! {
                deke_wreck::CollisionBody::new(
                    None,
                    deke_wreck::CollisionFilter {
                        links: [false; #n_lit],
                        ee: false,
                        base: false,
                        obstacles: true,
                    },
                    Vec::new(),
                    #n_joints as i16 + 1,
                )
            };
        }
    } else {
        ee_collider_build = quote! {
            deke_wreck::CollisionBody::new(
                None,
                deke_wreck::CollisionFilter {
                    links: [false; #n_lit],
                    ee: false,
                    base: false,
                    obstacles: false,
                },
                Vec::new(),
                #n_joints as i16 + 1,
            )
        };
    }

    // Base link — static collision body (never FK-transformed)
    let base_collider_build;
    if let Some(&base_idx) = link_map.get(base_link_name.as_str()) {
        let base_link = &links[base_idx];
        if !base_link.shapes.is_empty() {
            let (const_defs, build_stmts) =
                gen_link_shapes("BASE_LINK", &base_link.shapes, sphere_padding);
            all_const_defs.extend(const_defs);

            let (base_filter_links, base_filter_ee, base_filter_obstacles, _) = build_filter(
                n_joints + 1,
                n_joints,
                &joint_child_names,
                ee_link_name,
                base_link_name,
                &disabled_pairs,
                &forced_ee,
                &ignored_env,
            );

            let base_filter_bools: Vec<TokenStream2> = base_filter_links
                .iter()
                .map(|b| {
                    if *b {
                        quote! { true }
                    } else {
                        quote! { false }
                    }
                })
                .collect();

            base_collider_build = quote! {
                Some(deke_wreck::CollisionBody::new(
                    {
                        let mut collider = wreck::Collider::default();
                        #(#build_stmts)*
                        Some(collider)
                    },
                    deke_wreck::CollisionFilter {
                        links: [#(#base_filter_bools),*],
                        ee: #base_filter_ee,
                        base: false,
                        obstacles: #base_filter_obstacles,
                    },
                    Vec::new(),
                    -1i16,
                ))
            };
        } else {
            base_collider_build = quote! { None };
        }
    } else {
        base_collider_build = quote! { None };
    }

    let joint_name_strs: Vec<String> = joints.iter().map(|j| j.name.clone()).collect();
    let link_name_strs: Vec<&str> = joint_child_names.iter().map(|s| s.as_str()).collect();

    let inlined_tokens = if config.sphere_only {
        for joint in &joints {
            let child_link_idx = link_map[&joint.child];
            let link = &links[child_link_idx];
            for shape in &link.shapes {
                if !matches!(shape, ShapeData::Sphere { .. }) {
                    panic!(
                        "sphere_only is true but link '{}' has non-sphere collision shapes",
                        link.name
                    );
                }
            }
        }
        if !ee_is_joint_child {
            if let Some(&ee_idx) = ee_link_idx {
                for shape in &links[ee_idx].shapes {
                    if !matches!(shape, ShapeData::Sphere { .. }) {
                        panic!(
                            "sphere_only is true but end effector '{}' has non-sphere shapes",
                            links[ee_idx].name
                        );
                    }
                }
            }
        }
        if let Some(&base_idx) = link_map.get(base_link_name.as_str()) {
            for shape in &links[base_idx].shapes {
                if !matches!(shape, ShapeData::Sphere { .. }) {
                    panic!(
                        "sphere_only is true but base link '{}' has non-sphere shapes",
                        links[base_idx].name
                    );
                }
            }
        }

        let mut total_spheres = 0usize;
        for joint in &joints {
            let child_link_idx = link_map[&joint.child];
            total_spheres += links[child_link_idx].shapes.len();
        }
        if !ee_is_joint_child {
            if let Some(&ee_idx) = ee_link_idx {
                total_spheres += links[ee_idx].shapes.len();
            }
        }
        if let Some(&base_idx) = link_map.get(base_link_name.as_str()) {
            total_spheres += links[base_idx].shapes.len();
        }
        let total_spheres_lit = proc_macro2::Literal::usize_unsuffixed(total_spheres);

        let mut spheres_stmts = Vec::new();
        for (joint_idx, joint) in joints.iter().enumerate() {
            let child_link_idx = link_map[&joint.child];
            let link = &links[child_link_idx];
            if link.shapes.is_empty() {
                continue;
            }
            let arr_name = format_ident!("LINK_{}_SPHERES", joint_idx);
            let idx_lit = proc_macro2::Literal::usize_unsuffixed(joint_idx);
            spheres_stmts.push(quote! {
                for s in #arr_name.iter() {
                    let c = transforms[#idx_lit].transform_point3a(glam::Vec3A::from(s.center));
                    soa.push(wreck::Sphere::new(glam::Vec3::from(c), s.radius));
                }
            });
        }
        if !ee_is_joint_child {
            if let Some(&ee_idx) = ee_link_idx {
                if !links[ee_idx].shapes.is_empty() {
                    let last_idx = proc_macro2::Literal::usize_unsuffixed(n_joints - 1);
                    spheres_stmts.push(quote! {
                        for s in EE_SPHERES.iter() {
                            let c = transforms[#last_idx].transform_point3a(glam::Vec3A::from(s.center));
                            soa.push(wreck::Sphere::new(glam::Vec3::from(c), s.radius));
                        }
                    });
                }
            }
        }
        if let Some(&base_idx) = link_map.get(base_link_name.as_str()) {
            if !links[base_idx].shapes.is_empty() {
                spheres_stmts.push(quote! {
                    for s in BASE_LINK_SPHERES.iter() {
                        soa.push(*s);
                    }
                });
            }
        }

        let mut debug_soa_builds = Vec::new();
        for (joint_idx, joint) in joints.iter().enumerate() {
            let child_link_idx = link_map[&joint.child];
            let link = &links[child_link_idx];
            let idx_lit = proc_macro2::Literal::usize_unsuffixed(joint_idx);
            if !link.shapes.is_empty() {
                let arr_name = format_ident!("LINK_{}_SPHERES", joint_idx);
                debug_soa_builds.push(quote! {
                    {
                        let mut s = wreck::soa::SpheresSoA::new();
                        for sp in #arr_name.iter() {
                            let c = transforms[#idx_lit].transform_point3a(glam::Vec3A::from(sp.center));
                            s.push(wreck::Sphere::new(glam::Vec3::from(c), sp.radius));
                        }
                        s
                    }
                });
            } else {
                debug_soa_builds.push(quote! { wreck::soa::SpheresSoA::new() });
            }
        }

        let robot_name_str = &config.name;
        let n_minus_1 = proc_macro2::Literal::usize_unsuffixed(n_joints - 1);

        quote! {
            use deke_types::FKChain as _;

            const TOTAL_SPHERE_COUNT: usize = #total_spheres_lit;

            #[derive(Clone)]
            struct Inlined {
                fk: deke_types::URDFChain<#n_lit>,
                links: [deke_wreck::CollisionBody<#n_lit>; #n_lit],
                ee: deke_wreck::CollisionBody<#n_lit>,
                base: Option<deke_wreck::CollisionBody<#n_lit>>,
            }

            impl deke_wreck::InlinedRobot<#n_lit> for Inlined {
                fn name(&self) -> &str {
                    #robot_name_str
                }

                fn clone_box(&self) -> Box<dyn deke_wreck::InlinedRobot<#n_lit>> {
                    Box::new(self.clone())
                }

                fn validate(
                    &mut self,
                    q: deke_types::SRobotQ<#n_lit>,
                    env: &wreck::Collider,
                    ee_attachments: &[wreck::Collider],
                ) -> deke_types::DekeResult<()> {
                    let transforms = self.fk.fk(&q).map_err(|e| -> deke_types::DekeError { e.into() })?;

                    for i in 0..#n_minus_1 {
                        self.links[i].apply_transform(transforms[i]);
                        if let Some(ref ci) = self.links[i].base {
                            if self.links[i].filter.obstacles && ci.collides_other(env) {
                                return Err(deke_types::DekeError::EnvironmentCollision(
                                    self.links[i].token, self.links[i].token,
                                ));
                            }
                            for j in 0..i {
                                if !self.links[j].filter.links[i] || !self.links[i].filter.links[j] {
                                    continue;
                                }
                                if let Some(ref cj) = self.links[j].base {
                                    if ci.collides_other(cj) {
                                        return Err(deke_types::DekeError::SelfCollison(
                                            self.links[j].token, self.links[i].token,
                                        ));
                                    }
                                }
                            }
                            if let Some(ref base_body) = self.base {
                                if self.links[i].filter.base && base_body.filter.links[i] {
                                    if let Some(ref cb) = base_body.base {
                                        if ci.collides_other(cb) {
                                            return Err(deke_types::DekeError::SelfCollison(
                                                self.links[i].token, base_body.token,
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let last_tf = transforms[#n_minus_1];
                    let delta = self.links[#n_minus_1].compute_delta(last_tf);
                    self.links[#n_minus_1].apply_precomputed(delta, last_tf);
                    if let Some(ref cn) = self.links[#n_minus_1].base {
                        if self.links[#n_minus_1].filter.obstacles && cn.collides_other(env) {
                            return Err(deke_types::DekeError::EnvironmentCollision(
                                self.links[#n_minus_1].token, self.links[#n_minus_1].token,
                            ));
                        }
                        for j in 0..#n_minus_1 {
                            if !self.links[j].filter.links[#n_minus_1] || !self.links[#n_minus_1].filter.links[j] {
                                continue;
                            }
                            if let Some(ref cj) = self.links[j].base {
                                if cn.collides_other(cj) {
                                    return Err(deke_types::DekeError::SelfCollison(
                                        self.links[j].token, self.links[#n_minus_1].token,
                                    ));
                                }
                            }
                        }
                        if let Some(ref base_body) = self.base {
                            if self.links[#n_minus_1].filter.base && base_body.filter.links[#n_minus_1] {
                                if let Some(ref cb) = base_body.base {
                                    if cn.collides_other(cb) {
                                        return Err(deke_types::DekeError::SelfCollison(
                                            self.links[#n_minus_1].token, base_body.token,
                                        ));
                                    }
                                }
                            }
                        }
                    }

                    self.ee.apply_precomputed(delta, last_tf);
                    if let Some(ref ce) = self.ee.base {
                        if self.ee.filter.obstacles && ce.collides_other(env) {
                            return Err(deke_types::DekeError::EnvironmentCollision(
                                self.ee.token, self.ee.token,
                            ));
                        }
                        for j in 0..#n_lit {
                            if !self.links[j].filter.ee || !self.ee.filter.links[j] {
                                continue;
                            }
                            if let Some(ref cj) = self.links[j].base {
                                if ce.collides_other(cj) {
                                    return Err(deke_types::DekeError::SelfCollison(
                                        self.links[j].token, self.ee.token,
                                    ));
                                }
                            }
                        }
                        if let Some(ref base_body) = self.base {
                            if self.ee.filter.base && base_body.filter.ee {
                                if let Some(ref cb) = base_body.base {
                                    if ce.collides_other(cb) {
                                        return Err(deke_types::DekeError::SelfCollison(
                                            self.ee.token, base_body.token,
                                        ));
                                    }
                                }
                            }
                        }
                    }

                    for att in ee_attachments {
                        if att.collides_other(env) {
                            return Err(deke_types::DekeError::EnvironmentCollision(-2, -2));
                        }
                    }

                    Ok(())
                }

                fn spheres(&self, q: deke_types::SRobotQ<#n_lit>) -> wreck::soa::SpheresSoA {
                    let transforms = self.fk.fk(&q).unwrap();
                    let mut soa = wreck::soa::SpheresSoA::with_capacity(TOTAL_SPHERE_COUNT);
                    #(#spheres_stmts)*
                    soa
                }

                fn eefk(&self, q: deke_types::SRobotQ<#n_lit>) -> glam::Affine3A {
                    self.fk.fk_end(&q).unwrap()
                }

                fn debug_self_collisions(&self, q: deke_types::SRobotQ<#n_lit>) -> Vec<(usize, usize)> {
                    let transforms = self.fk.fk(&q).unwrap();
                    let link_soas = [#(#debug_soa_builds),*];
                    let mut pairs = Vec::new();
                    for i in 0..#n_lit {
                        for j in 0..i {
                            if link_soas[j].any_collides_soa(&link_soas[i]) {
                                pairs.push((j, i));
                            }
                        }
                    }
                    pairs
                }
            }

            pub type InlinedValidator = deke_types::ValidatorAnd<
                deke_types::JointValidator<#n_lit>,
                deke_wreck::InlinedWreckValidator<#n_lit>,
            >;

            pub fn inlined_validator(environment: wreck::Collider) -> InlinedValidator {
                let fk = deke_types::URDFChain::<#n_lit>::new(URDF_JOINTS);
                let links = [#(#link_collider_builds),*];
                let ee = #ee_collider_build;
                let base = #base_collider_build;
                let robot = Inlined { fk, links, ee, base };
                let inlined = deke_wreck::InlinedWreckValidator::new(Box::new(robot), environment);
                let joints = deke_types::JointValidator::new(
                    deke_types::SRobotQ(JOINT_LOWER),
                    deke_types::SRobotQ(JOINT_UPPER),
                );
                deke_types::ValidatorAnd(joints, inlined)
            }
        }
    } else {
        quote! {}
    };

    let output = quote! {
        pub mod #mod_name {
            #(#all_const_defs)*

            pub const DOF: usize = #n_lit;
            pub const JOINT_NAMES: [&str; #n_lit] = [#(#joint_name_strs),*];
            pub const LINK_NAMES: [&str; #n_lit] = [#(#link_name_strs),*];
            pub const END_EFFECTOR: &str = #ee_link_name;

            pub const JOINT_LOWER: [f32; #n_lit] = [#(#lower_vals),*];
            pub const JOINT_UPPER: [f32; #n_lit] = [#(#upper_vals),*];
            pub const JOINT_VELOCITY: [f32; #n_lit] = [#(#velocity_vals),*];

            pub const DH_JOINTS: [deke_types::DHJoint; #n_lit] = [
                #(#dh_joint_tokens),*
            ];

            pub const URDF_JOINTS: [deke_types::URDFJoint; #n_lit] = [
                #(#urdf_joint_tokens),*
            ];

            pub type CollisionValidator = deke_wreck::WreckValidator<
                #n_lit,
                deke_types::URDFChain<#n_lit>,
            >;

            pub type Validator = deke_types::ValidatorAnd<
                deke_types::JointValidator<#n_lit>,
                CollisionValidator,
            >;

            pub fn validator(environment: wreck::Collider) -> Validator {
                let fk = deke_types::URDFChain::<#n_lit>::new(URDF_JOINTS);

                let links = [#(#link_collider_builds),*];
                let ee = #ee_collider_build;
                let base = #base_collider_build;

                let joints = deke_types::JointValidator::new(
                    deke_types::SRobotQ(JOINT_LOWER),
                    deke_types::SRobotQ(JOINT_UPPER),
                );
                let collisions = deke_wreck::WreckValidator::new(links, ee, base, environment, fk);

                deke_types::ValidatorAnd(joints, collisions)
            }

            pub fn rrtc(settings: deke_rrt::RrtcSettings<#n_lit>) -> deke_rrt::RrtcPlanner<#n_lit> {
                deke_rrt::RrtcPlanner::new(settings)
            }

            pub fn aorrtc(settings: deke_rrt::AorrtcSettings<#n_lit>) -> deke_rrt::AorrtcPlanner<#n_lit> {
                deke_rrt::AorrtcPlanner::new(settings)
            }

            pub fn krrtc(settings: deke_rrt::KrrtcSettings<#n_lit>) -> deke_rrt::KrrtcPlanner<#n_lit> {
                deke_rrt::KrrtcPlanner::new(settings)
            }

            #inlined_tokens
        }
    };

    output.into()
}

enum ValidatorExpr {
    Atom(Expr),
    Not(Box<ValidatorExpr>),
    And(Box<ValidatorExpr>, Box<ValidatorExpr>),
    Or(Box<ValidatorExpr>, Box<ValidatorExpr>),
}

impl ValidatorExpr {
    fn parse_or(input: ParseStream) -> syn::Result<Self> {
        let mut left = Self::parse_and(input)?;
        while input.peek(Token![|]) {
            input.parse::<Token![|]>()?;
            let right = Self::parse_and(input)?;
            left = ValidatorExpr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(input: ParseStream) -> syn::Result<Self> {
        let mut left = Self::parse_unary(input)?;
        while input.peek(Token![&]) {
            input.parse::<Token![&]>()?;
            let right = Self::parse_unary(input)?;
            left = ValidatorExpr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_unary(input: ParseStream) -> syn::Result<Self> {
        if input.peek(Token![!]) {
            input.parse::<Token![!]>()?;
            let inner = Self::parse_unary(input)?;
            Ok(ValidatorExpr::Not(Box::new(inner)))
        } else if input.peek(syn::token::Paren) {
            let content;
            syn::parenthesized!(content in input);
            Self::parse_or(&content)
        } else {
            let expr: Expr = input.parse()?;
            Ok(ValidatorExpr::Atom(expr))
        }
    }

    fn to_tokens(&self) -> TokenStream2 {
        match self {
            ValidatorExpr::Atom(expr) => quote! { #expr },
            ValidatorExpr::Not(inner) => {
                let inner = inner.to_tokens();
                quote! { deke_types::ValidatorNot(#inner) }
            }
            ValidatorExpr::And(left, right) => {
                let left = left.to_tokens();
                let right = right.to_tokens();
                quote! { deke_types::ValidatorAnd(#left, #right) }
            }
            ValidatorExpr::Or(left, right) => {
                let left = left.to_tokens();
                let right = right.to_tokens();
                quote! { deke_types::ValidatorOr(#left, #right) }
            }
        }
    }
}

struct CombineValidatorsInput(ValidatorExpr);

impl Parse for CombineValidatorsInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        ValidatorExpr::parse_or(input).map(CombineValidatorsInput)
    }
}

/// Combine validators using boolean logic.
///
/// `&` means AND (both must pass), `|` means OR (either passes),
/// `!` means NOT (inverts the result). Parentheses control precedence.
///
/// ```ignore
/// let v = combine_validators!(collision_check & limit_check);
/// let v = combine_validators!(zone_a | zone_b);
/// let v = combine_validators!((primary & !exclusion) | fallback);
/// ```
#[proc_macro]
pub fn combine_validators(input: TokenStream) -> TokenStream {
    let parsed = syn::parse_macro_input!(input as CombineValidatorsInput);
    parsed.0.to_tokens().into()
}
