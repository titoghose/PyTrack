Introduction
=================

This is a toolkit to analyse and visualize eye tracking data. It
provides the following functionality:

Feature Extraction
------------------

This involves extraction of parameters or meta-data related to blinks,
fixations, saccades, microsaccades and pupil diameter. The features
extracted are as follows:

============ ============ ========= ============= ================ ====================
Blink        Fixations    Saccades  Microsaccades Pupil            Revisits to AOI/ROI
============ ============ ========= ============= ================ ====================
Count        Count        Count     Count         Size             Count
Avg Duration Avg Duration Velocity  Velocity      Time to Peak     First Pass Duration
Max Duration Max Duration Amplitude Amplitude     Peak Size        Second Pass Duration
\                         Duration  Duration      Avg Size
\                                                 Slope
\                                                 Area Under Curve
============ ============ ========= ============= ================ ====================

Statistical Analysis
--------------------

After extraction of features, PyTrack can perform tests such as the
student T-Test, Welch T-Test, ANOVA, RMANOVA, n-way ANOVA and Mixed
ANOVA. The between and within group factors can be specified.

Visualization
-------------

PyTrack can generate a variety of plots. The visualization is through an
interactive GUI. The plots that can be generated are as follows:

1. Fixation plot
2. Individual subject gaze heat map
3. Aggregate subject gaze heat map
4. Dynamic pupil size and gaze plot
5. Microsaccade position and velocity plot
6. Microsaccade main sequence plot
