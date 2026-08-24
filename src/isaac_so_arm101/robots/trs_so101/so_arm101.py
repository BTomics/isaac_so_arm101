from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

TEMPLATE_ASSETS_DATA_DIR = Path(__file__).resolve().parent

# Reflected rotor inertia, kg m^2, seen at the joint: I_rotor * N^2.
#
# WHY IT IS HERE AT ALL: a rollout of the A' policy recorded wrist_flex 1.073 rad
# (61 deg) PAST its hard limit on 23.9% of steps, while the drive never commanded
# a target below that limit - 0.0% of commanded targets were below it. So the
# joint was being forced through its stop by contact, and PhysX was not holding
# the constraint. A joint with no armature is the standard way that happens: with
# zero rotor inertia the effective mass at a light distal link is tiny, and a
# stiff implicit drive plus a contact impulse wins against a position constraint
# solved in a finite number of iterations.
#
# THE NUMBER IS AN ESTIMATE, not a measurement. The STS3215's reduction is ~1:345,
# so N^2 ~ 1.2e5, and small DC rotors run 1e-7..1e-6 kg m^2 - which brackets the
# reflected value at 0.012..0.12. 0.02 sits at the low end of that bracket
# deliberately: too much armature makes the arm sluggish in a way the real servo
# is not, and the failure it is here to fix shows up long before the top of the
# range. VALIDATE IT with scripts/limit_probe.py rather than trusting the
# arithmetic - the bracket spans a factor of ten.
ARMATURE = 0.02

##
# Configuration
##

SO_ARM101_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        replace_cylinders_with_capsules=True,
        asset_path=f"{TEMPLATE_ASSETS_DATA_DIR}/urdf/so_arm101.urdf",
        activate_contact_sensors=False, # set as false while waiting for capsule implementation
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            # 5.0 m/s of depenetration can hurl a light distal joint through its
            # stop when a capsule approximation interpenetrates in a folded pose.
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            # Joint limits are POSITION constraints. 8 iterations was thin for a
            # stiff implicit drive against a contact; 32 is what holds them.
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": -0.0,
            "wrist_flex": 1.57,
            "wrist_roll": -0.0,
            "gripper": 0.0,
        },
        # Set initial joint velocities to zero
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Shoulder Pan      moves: ALL masses                   (~0.8kg total)
        # Shoulder Lift     moves: Everything except base       (~0.65kg)
        # Elbow             moves: Lower arm, wrist, gripper    (~0.38kg)
        # Wrist Pitch       moves: Wrist and gripper            (~0.24kg)
        # Wrist Roll        moves: Gripper assembly             (~0.14kg)
        # Jaw               moves: Only moving jaw              (~0.034kg)
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_flex", "wrist_.*"],
            effort_limit_sim=1.9,
            velocity_limit_sim=1.5,
            armature=ARMATURE,
            stiffness={
                "shoulder_pan": 200.0,  # Highest - moves all mass
                "shoulder_lift": 170.0,  # Slightly less than rotation
                "elbow_flex": 120.0,  # Reduced based on less mass
                "wrist_flex": 80.0,  # Reduced for less mass
                "wrist_roll": 50.0,  # Low mass to move
            },
            damping={
                "shoulder_pan": 80.0,
                "shoulder_lift": 65.0,
                "elbow_flex": 45.0,
                "wrist_flex": 30.0,
                "wrist_roll": 20.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=2.5,  # Increased from 1.9 to 2.5 for stronger grip
            velocity_limit_sim=1.5,
            armature=ARMATURE,
            stiffness=60.0,  # Increased from 25.0 to 60.0 for more reliable closing
            damping=20.0,  # Increased from 10.0 to 20.0 for stability
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)

