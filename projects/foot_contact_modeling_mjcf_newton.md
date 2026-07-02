# Foot Contact Modeling for MJCF + Newton

This note captures a practical path for building a more realistic foot-ground contact model for the kinodynamic human retargeting project. The immediate goal is an MJCF asset that can be used in ProtoMotions-style controller/RL workflows, while leaving a path toward Newton SDF/hydroelastic contact for more physically distributed plantar contact.

## Why not a single point contact?

A running human foot does not contact the ground at one point. During stance the load moves through a spatial patch: often heel strike, midfoot loading, forefoot loading, and toe-off. For treadmill running, we also care about braking/propulsive shear, center-of-pressure migration, torsional foot-ground moments, and slip relative to the moving belt.

A single point or a single small collision primitive tends to produce:

- noisy or unstable ground reaction forces,
- unrealistic foot rocking,
- poor center-of-pressure behavior,
- excessive slip or artificial stickiness,
- bad gradients/objectives for kinodynamic fitting,
- RL policies that exploit contact artifacts.

## Survey of useful contact modeling approaches

### 1. Multi-sphere compliant foot contact

Common in biomechanics and trajectory optimization: place multiple contact spheres under the calcaneus, metatarsal heads, lateral foot, medial foot, and toes. Each sphere uses a compliant normal law and friction model.

Relevant analogs:

- OpenSim `HuntCrossleyForce`: Hertz/Hunt-Crossley sphere-halfspace contact.
- OpenSim `SmoothSphereHalfSpaceForce`: differentiable sphere-halfspace contact with static, dynamic, and viscous friction smoothing.

Pros:

- simple,
- easy to tune per anatomical region,
- works well for optimization,
- gives a coarse plantar pressure distribution.

Cons:

- still a finite set of localized contacts,
- spherical contacts can feel too rounded unless many spheres are used,
- pure MJCF cannot directly express a custom Hunt-Crossley law; it approximates this through MuJoCo/Newton contact parameters.

### 2. Multi-segment ellipsoid MJCF foot

This is the current preferred MJCF-compatible MVP. Instead of one foot collision box or a grid of rectangular patches, use an articulated anatomical foot with several bodies:

- hindfoot / calcaneus,
- midfoot,
- forefoot / metatarsal segment,
- toes.

Connect the segments with compliant hinge joints and use named ellipsoid contact geoms for anatomical load regions:

- calcaneus,
- navicular / medial midfoot,
- cuboid / lateral midfoot,
- first metatarsal head,
- central metatarsal heads,
- fifth metatarsal head,
- hallux,
- lesser toes.

Pros:

- still easy to express in MJCF,
- smoother rolling behavior than flat boxes,
- better anatomical correspondence for treadmill force/pressure data,
- explicit named geoms allow per-region friction/contact tuning,
- segment joints allow heel-to-toe progression and toe-off mechanics,
- contact/touch sensors can approximate plantar load distribution.

Cons:

- MuJoCo contact is still point-contact based internally,
- each ellipsoid-plane pair usually contributes one contact point,
- geometry and keyframe heights must be tuned to avoid excess penetration,
- not a true continuum pressure model.

### 3. Toe segment + contact patch model

Use a separate toe body/joint with its own forefoot and toe contact geoms. This matters for running because toe-off mechanics strongly affect horizontal impulse and trunk angular momentum.

The existing `smpl_humanoid.xml` already has ankle and toe bodies, making it a better starting point than `amp_humanoid.xml`, whose foot is a single box with `condim="1"`.

Recommended structure:

- ankle/foot body: heel + midfoot + proximal forefoot patches,
- toe body: distal forefoot + toe pad patches,
- tune toe joint stiffness/damping and joint limits if using actuated/dynamic toes.

### 4. Elastic foundation / mesh contact concept

OpenSim `ElasticFoundationForce` approximates distributed contact by placing springs at contact mesh face centers. Conceptually this is closer to plantar pressure: contact force is integrated over a surface rather than generated at one or a few points.

Pros:

- better pressure distribution,
- can use foot sole shape more directly,
- maps well to instrumented treadmill pressure/force data.

Cons:

- not directly available as pure MJCF primitive contact,
- MuJoCo mesh collision has limitations and convexification concerns,
- can be more expensive and harder to tune.

### 5. MuJoCo contact improvements for MJCF

Even though MuJoCo contacts are point contacts, MJCF can approximate contact patches with better solver/contact settings:

```xml
<option timestep="0.001"
        integrator="implicitfast"
        solver="Newton"
        cone="elliptic"
        impratio="10"
        iterations="50"
        tolerance="1e-10"
        noslip_iterations="2">
  <flag nativeccd="enable" multiccd="enable"/>
</option>
```

Recommended foot-contact settings:

```xml
<geom class="foot_contact"
      type="box"
      condim="6"
      friction="1.1 0.03 0.002"
      solref="0.012 1"
      solimp="0.90 0.98 0.002 0.5 2"
      margin="0.001"/>
```

Use `condim="4"` or `condim="6"` rather than `condim="1"`:

- `condim=3`: normal + tangential friction,
- `condim=4`: adds torsional friction around the contact normal,
- `condim=6`: adds torsional and rolling friction.

For explicit foot-floor pairs, MJCF supports a five-value friction vector:

```xml
<contact>
  <pair geom1="L_heel_medial" geom2="floor"
        condim="6"
        friction="1.2 0.9 0.03 0.002 0.002"
        solref="0.012 1"
        solimp="0.90 0.98 0.002 0.5 2"/>
</contact>
```

The five values are:

1. tangential friction direction 1,
2. tangential friction direction 2,
3. torsional friction,
4. rolling friction direction 1,
5. rolling friction direction 2.

This is useful for treadmill running because the belt direction and cross-belt direction can be tuned separately if the contact frame is controlled/understood.

### 6. Newton SDF / hydroelastic long-term direction

Newton has a stronger path to true distributed contact via SDF and hydroelastic contact. Instead of relying only on MuJoCo-generated contacts, we can use Newton collision generation and feed contacts into a solver.

Conceptually:

```python
pipeline = newton.CollisionPipeline(model, broad_phase="sap")
solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False)
contacts = pipeline.contacts()
pipeline.collide(state_0, contacts)
solver.step(state_0, state_1, control, contacts, dt)
```

Newton material/shape fields to investigate for feet and treadmill:

- `is_hydroelastic=True`,
- `kh` for hydroelastic stiffness,
- `sdf_max_resolution`,
- `mu`,
- `ke`, `kd`, `kf`,
- `mu_torsional`,
- `mu_rolling`.

This likely becomes a Python-side Newton model/collision-pipeline configuration rather than a pure MJCF-only feature. The MJCF MVP should therefore be designed with named, anatomically meaningful foot patches that can later map cleanly onto SDF or hydroelastic sole regions.

## Recommended MVP

Start from `protomotions/data/assets/mjcf/smpl_humanoid.xml` because it already has separate ankle and toe bodies. Do not mutate it directly at first. Copy it to a new asset such as:

```text
projects/humanoid_model/smpl_humanoid_foot_contact.xml
```

Then replace the existing ankle/toe collision boxes with named contact patches.

For each side, prefer anatomical ellipsoid contacts:

```text
L_calcaneus_contact
L_navicular_contact
L_cuboid_contact
L_metatarsal_1_contact
L_metatarsal_2_4_contact
L_metatarsal_5_contact
L_hallux_contact
L_lesser_toes_contact
```

and the same for `R_*`.

Use:

- `condim="6"` for foot patches,
- elliptic friction cones,
- Newton solver,
- `multiccd` for better primitive contact manifolds,
- explicit foot-floor contact pairs,
- touch/contact sensors for each patch or grouped region,
- treadmill-relative slip metrics.

## Initial geometry layout

Approximate each foot in a local frame where:

- `+x` is forward toward the toes,
- `+y` is medial/lateral depending on side,
- `+z` is upward,
- ellipsoid contact lobes sit slightly below each segment visual body.

Suggested articulated layout for a generic adult foot:

| Segment | Contact geom | x center | y center | radius x | radius y | radius z |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hindfoot | calcaneus | -0.058 | 0.000 | 0.052 | 0.038 | 0.018 |
| midfoot | navicular | 0.022 | +0.016 | 0.030 | 0.018 | 0.016 |
| midfoot | cuboid | 0.024 | -0.018 | 0.036 | 0.020 | 0.016 |
| forefoot | metatarsal 1 | 0.030 | +0.030 | 0.032 | 0.020 | 0.016 |
| forefoot | metatarsals 2-4 | 0.038 | 0.000 | 0.040 | 0.025 | 0.017 |
| forefoot | metatarsal 5 | 0.030 | -0.030 | 0.032 | 0.020 | 0.016 |
| toes | hallux | 0.030 | +0.024 | 0.038 | 0.018 | 0.014 |
| toes | lesser toes | 0.030 | -0.017 | 0.038 | 0.026 | 0.014 |

These are starting values only. They should be scaled to subject-specific foot length/width from mocap calibration, scans, or marker-derived foot landmarks.

## Parameters to tune

### Contact/friction

- tangential friction: start around `0.9-1.3`,
- anisotropic treadmill friction: tune belt direction separately from cross-belt direction,
- torsional friction: start around `0.02-0.05`,
- rolling friction: start around `0.001-0.005`,
- `condim`: compare `4` vs `6`.

### Compliance

- `solref`: start around `0.008-0.02 1`,
- `solimp`: start around `0.90 0.98 0.001-0.003 0.5 2`,
- `margin`: start around `0.0005-0.002`.

### Solver

- timestep: `0.001` for contact-heavy validation,
- solver: `Newton`,
- cone: `elliptic`,
- `impratio`: `5-20`,
- `iterations`: `50-100`,
- `noslip_iterations`: `0-5`, compare for slip reduction vs cost.

## Metrics for validation

For each trial pose or motion segment, record:

- total vertical GRF,
- anterior-posterior and medial-lateral shear,
- center of pressure from patch forces,
- contact patch activation sequence,
- foot slip relative to treadmill belt velocity,
- free moment / yaw stability,
- solver iterations and contact jitter,
- whether the controller exploits edge contacts.

For instrumented treadmill data, compare simulated GRF and center-of-pressure trajectories against measured treadmill signals in the treadmill frame.

## Anatomical foot-model survey update

After reviewing OpenSim/Moco, multisegment foot-model, and ground-contact-personalization references, the more anatomical direction should combine two ideas:

1. **Multisegment foot kinematics** from clinical gait analysis.
2. **Distributed plantar contact** from contact spheres/ellipsoids or elastic-foundation grids.

### Multisegment foot models

Several common multisegment foot models do not treat the foot as one rigid body:

| Model family | Typical segments | Useful lesson for MJCF |
| --- | --- | --- |
| Oxford Foot Model | shank, rearfoot, forefoot, hallux | Separate hallux/toe segment matters, but midfoot is not explicit. |
| Milwaukee Foot Model | shank, rearfoot, forefoot including distal tarsals, hallux | Segment definitions change joint kinetics. |
| Vogel-style model | shank, rearfoot, midfoot, forefoot, phalanges | Good compact structure for dynamic arch + toe-off. |
| Ghent Foot Model | shank, rearfoot, midfoot, medial forefoot, lateral forefoot, hallux | Medial/lateral forefoot split is useful for COP and edge loading. |
| Maharaj/Rainbow/Lichtwark-style OpenSim model | tibia, talus, calcaneus, midfoot, forefoot, toes | More anatomical ankle-foot complex; includes talus/calcaneus separation and multiple internal joints. |

Recent comparisons of multisegment foot models emphasize that segment definitions affect tibiotalar, midtarsal, and MTP joint moments/powers. More segments are not automatically better because small segments are hard to track with skin markers, but a separate midfoot segment better captures dynamic arch behavior than a rigid foot.

### Contact modeling patterns

The literature and tools use several contact patterns:

- **OpenSim/Moco smooth sphere-halfspace contact**: differentiable compliant contact between contact spheres and ground. Moco examples often group heel and forefoot contact elements and track summed GRF against experimental force plate data.
- **Hunt-Crossley contact spheres**: common for heel/front/toe approximations, sometimes 2-3 spheres per foot in gait prediction or AFO studies.
- **Elastic foundation contact**: surface/grid of springs across the plantar foot; better matches pressure/GRF distribution but is heavier and less MJCF-native.
- **NMSM Ground Contact Personalization**: calibrates elastic foundation foot-ground contact from IK + GRF data. It uses a uniform grid of springs with nonlinear damping/friction across the bottom of the foot and can tune stiffness distribution, friction, belt speed, and electrical center/COP terms.
- **Foot-ground contact modeling review**: common geometries include points, circles, ellipses, spheres, ellipsoids, rectangular contact elements, and surfaces from 3D scans. There is no universal standard procedure.

### Implications for this project

For a Newton/MJCF sandbox, the next model should be a **hybrid of Vogel + Ghent + simplified OpenSim ankle-foot anatomy**:

```text
shank / ankle frame
  talus
    calcaneus / rearfoot
      midfoot
        medial_forefoot
          hallux
        lateral_forefoot
          lesser_toes
```

For the one-foot sandbox we can omit the shank dynamics initially, but the segment names and joint layout should match what we will eventually plug into `projects/humanoid_model/smpl_humanoid_foot_contact.xml`.

Recommended MJCF bodies:

```text
talus_root
calcaneus
midfoot
medial_forefoot
lateral_forefoot
hallux
lesser_toes
```

Recommended joints:

```text
subtalar_roll             talus -> calcaneus
midtarsal_pitch           calcaneus -> midfoot
midtarsal_roll            calcaneus -> midfoot
tmt_medial_pitch          midfoot -> medial_forefoot
tmt_lateral_pitch         midfoot -> lateral_forefoot
mtp_hallux_pitch          medial_forefoot -> hallux
mtp_lesser_toes_pitch     lateral_forefoot -> lesser_toes
```

Recommended anatomical contact lobes:

```text
heel_medial_pad
heel_lateral_pad
lateral_midfoot_pad       # cuboid / lateral arch; often contacts more than medial arch
medial_arch_pad           # higher/softer, may contact only under large load or flat-foot
met_head_1_pad
met_head_2_3_pad
met_head_4_5_pad
hallux_pad
lesser_toes_pad
```

Important: the medial arch should not be a big flat load-bearing patch by default. In normal foot mechanics, the lateral midfoot/cuboid side is more likely to contact the ground, while the medial arch should be higher and softer unless the subject has a low arch, shoe compression, or large load.

### Subject-specific anatomy inputs

To make this subject-specific from treadmill mocap, derive or fit:

- heel marker / calcaneus posterior-inferior point,
- toe marker,
- medial/lateral forefoot markers,
- first and fifth metatarsal landmarks if available,
- foot length and width,
- arch height or navicular/midfoot marker height if available,
- treadmill belt speed and force-plate COP/GRF.

Then scale:

- segment lengths,
- ellipsoid radii,
- plantar lobe positions,
- stiffness/friction per lobe,
- arch/tarsometatarsal/MTP joint stiffness.

### Better calibration target

Instead of only eyeballing contacts, calibrate lobe parameters against instrumented treadmill outputs:

- total GRF,
- AP/ML shear,
- center of pressure trajectory,
- free moment / yaw torque,
- stance timing,
- treadmill-relative slip,
- per-region pressure if the treadmill provides pressure-map data.

This is analogous to NMSM Ground Contact Personalization, but using MJCF ellipsoid lobes first and Newton hydroelastic/SDF later.

## Park/Yu/Lee-inspired capsule sandbox

`projects/feet_models/bones_foot.xml` now implements a clean-room approximation of Park, Yu, and Lee's multi-segment capsule foot model. I did not find an official XML/model download on the public project page, so this MJCF is reconstructed from the paper description rather than copied from an author asset. `projects/feet_models/foot_contact_sandbox.xml` is kept as the working sandbox copy.

Current sandbox structure:

- 16 artificial bone primitives represented as MJCF capsules.
- 5 controlled segments:
  - `medial_metatarsal`,
  - `lateral_metatarsal`,
  - `heel`,
  - `medial_phalanges`,
  - `lateral_phalanges`.
- 4 ball joints / 12 internal foot DoFs:
  - `heel_ball`,
  - `lateral_metatarsal_ball`,
  - `lateral_phalanges_ball`,
  - `medial_phalanges_ball`.
- Foot-floor contacts use `condim="6"`, elliptic cones, explicit contact pairs, and named regional touch sensors.
- Foot bones collide with the floor but not with each other, avoiding internal capsule self-contact chatter.

The keyframes are intended as contact-regime tests rather than exact gait poses:

| Keyframe | Intended contact behavior |
| --- | --- |
| `rest_pose` | broad heel + lateral metatarsal + toe capsule support |
| `heel_strike` | calcaneus pair only |
| `tiptoe` | toe / distal metatarsal loading, heel lifted |
| `inside_tilt` | medial toe-edge loading |
| `outside_tilt` | lateral fifth-toe edge loading |

This model is closer to the paper's contact strategy than the earlier ellipsoid/box-patch variants because capsule endpoints and capsule-plane manifolds produce multiple, anatomically named contact opportunities while keeping the MJCF compact.

## Brown/McPhee ellipsoid foot sandbox

`projects/feet_models/elipsoid_feet.xml` implements a clean-room MJCF proxy for Brown and McPhee's "A 3D ellipsoidal volumetric foot-ground contact model for forward dynamics". The filename/model name keep the requested `elipsoid_feet` spelling, while the MJCF comments use the paper's standard `ellipsoid` terminology.

Current sandbox structure:

- 2 rigid segments:
  - `foot_segment`, containing the heel and ball-of-foot contact ellipsoids,
  - `toe_segment`, connected to the foot by `toe_hinge`.
- Hybrid contact geometry based on the paper's three-contact layout:
  - `heel_ellipsoid`,
  - `ball_ellipsoid`,
  - `toe_capsule`.
- The heel and ball ellipsoid radii, centroids, and rotations use the paper's Table 3 initial-guess parameters, converted from millimetres to metres. The toe contact is a capsule, pulled farther anterior from the midfoot and angled with the ball/midfoot contact direction.
- The model uses the paper/Table 3 frame convention: `+x` medial, `+y` anterior toward the toes, `+z` upward.
- A toe hinge about the medial-lateral `x` axis, placed about 16 mm above the ground following the paper's calibration description.
- Foot-floor contacts use `condim="6"`, elliptic cones, explicit contact pairs, and regional touch sensors.

Important limitation: MJCF cannot encode the paper's exact volumetric contact law, `F_n = k_V V (1 + a_V v_cn)`, directly in XML. This sandbox uses MuJoCo primitive geoms plus soft contact settings as a practical proxy. For subject-specific work, tune contact poses/sizes and pair `solref`/`solimp`/friction values against GRF, COP, and pressure data.

Inspect it with:

```bash
python examples/foot_contact_viewer.py --model-file projects/feet_models/elipsoid_feet.xml --paused --show-contacts
```

## Proposed implementation sequence

1. Build a standalone MJCF sandbox with one free foot and multiple sole patches.
2. Inspect it with:

   ```bash
   python examples/foot_contact_viewer.py --paused --show-contacts
   # or explicitly:
   python examples/foot_contact_viewer.py --model-file projects/feet_models/bones_foot.xml --paused --show-contacts
   ```

3. Validate flat-foot, heel-strike, forefoot, toe-off, and lateral-edge loading.
4. Copy `smpl_humanoid.xml` to `projects/humanoid_model/smpl_humanoid_foot_contact.xml`.
5. Replace the ankle/toe boxes with named patch geoms.
6. Add explicit foot-floor contact pairs and sensors.
7. Add treadmill frame conventions and slip metrics.
8. Integrate into kinodynamic retargeting and RL imitation workflows.
9. Revisit Newton SDF/hydroelastic foot soles once the MJCF patch baseline is stable.
