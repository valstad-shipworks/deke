//! Small enums representing 0..N solutions of an IK subproblem.

use arrayvec::ArrayVec;
use std::fmt::{Debug, Display, Formatter, Result};

#[derive(Debug, Clone)]
pub enum SolutionSet2<T> {
    Zero,
    One(T),
    Two(T, T),
}

#[derive(Debug, Clone)]
pub enum SolutionSet4<T> {
    Zero,
    One(T),
    Two(T, T),
    Three(T, T, T),
    Four(T, T, T, T),
}

impl<T: Copy> SolutionSet2<T> {
    pub fn expect_one(&self) -> T {
        match self {
            Self::Zero => panic!("Found no solutions where one was expected"),
            Self::One(s) => *s,
            Self::Two(..) => panic!("Found two solutions where one was expected"),
        }
    }

    pub fn expect_two(&self) -> (T, T) {
        match self {
            Self::Zero => panic!("Found no solutions where two were expected"),
            Self::One(_) => panic!("Found one solution where two were expected"),
            Self::Two(s1, s2) => (*s1, *s2),
        }
    }

    pub fn get_first(&self) -> T {
        match self {
            Self::Zero => panic!("No solutions"),
            Self::One(s) => *s,
            Self::Two(s, _) => *s,
        }
    }

    pub fn get_all(&self) -> ArrayVec<T, 2> {
        let mut v = ArrayVec::new();
        match self {
            Self::Zero => {}
            Self::One(s) => v.push(*s),
            Self::Two(s1, s2) => {
                v.push(*s1);
                v.push(*s2);
            }
        }
        v
    }

    pub fn duplicated(&self) -> Self {
        match self {
            SolutionSet2::One(s) => SolutionSet2::Two(*s, *s),
            SolutionSet2::Two(s1, s2) => SolutionSet2::Two(*s1, *s2),
            SolutionSet2::Zero => SolutionSet2::Zero,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            SolutionSet2::Zero => 0,
            SolutionSet2::One(_) => 1,
            SolutionSet2::Two(..) => 2,
        }
    }
}

impl<T: Copy + Debug> SolutionSet2<T> {
    pub fn from_vec(vec: &[T]) -> Self {
        match vec.len() {
            0 => Self::Zero,
            1 => Self::One(vec[0]),
            2 => Self::Two(vec[0], vec[1]),
            i => panic!("Vector {vec:?} contains too many solutions: {i}"),
        }
    }
}

impl<T: Copy> SolutionSet4<T> {
    pub fn get_all(&self) -> ArrayVec<T, 4> {
        let mut v = ArrayVec::new();
        match self {
            Self::Zero => {}
            Self::One(s) => v.push(*s),
            Self::Two(s1, s2) => {
                v.push(*s1);
                v.push(*s2);
            }
            Self::Three(s1, s2, s3) => {
                v.push(*s1);
                v.push(*s2);
                v.push(*s3);
            }
            Self::Four(s1, s2, s3, s4) => {
                v.push(*s1);
                v.push(*s2);
                v.push(*s3);
                v.push(*s4);
            }
        }
        v
    }
}

impl<T: Copy + Debug> SolutionSet4<T> {
    pub fn from_vec(vec: &[T]) -> Self {
        match vec.len() {
            0 => Self::Zero,
            1 => Self::One(vec[0]),
            2 => Self::Two(vec[0], vec[1]),
            3 => Self::Three(vec[0], vec[1], vec[2]),
            4 => Self::Four(vec[0], vec[1], vec[2], vec[3]),
            i => panic!("Vector {vec:?} contains too many solutions: {i}"),
        }
    }
}

impl<T: Display> Display for SolutionSet2<T> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            Self::Zero => write!(f, "{{ }}"),
            Self::One(s) => write!(f, "{{ {s} }}"),
            Self::Two(s1, s2) => write!(f, "{{ {s1} {s2} }}"),
        }
    }
}

impl<T: Display> Display for SolutionSet4<T> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            Self::Zero => write!(f, "{{ }}"),
            Self::One(s) => write!(f, "{{ {s} }}"),
            Self::Two(s1, s2) => write!(f, "{{ {s1} {s2} }}"),
            Self::Three(s1, s2, s3) => write!(f, "{{ {s1} {s2} {s3} }}"),
            Self::Four(s1, s2, s3, s4) => write!(f, "{{ {s1} {s2} {s3} {s4} }}"),
        }
    }
}
