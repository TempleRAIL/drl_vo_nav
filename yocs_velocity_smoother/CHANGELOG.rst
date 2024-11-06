^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package yocs_velocity_smoother
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.6.3 (2014-12-05)
------------------

0.6.2 (2014-11-30)
------------------
* yocs_velocity_smoother: adds node name param to launcher
* adds a little launcher restructing for muxer and smoother
* Contributors: Marcus Liebhardt

0.6.0 (2014-07-08)
------------------
* updating package informations. remove email for authors. updating maintainer
* Contributors: Jihoon Lee

0.5.3 (2014-03-24)
------------------

0.5.2 (2013-11-05)
------------------

0.5.1 (2013-10-14)
------------------
* Unify naming politics for binaries and plugins.

0.5.0 (2013-10-11)
------------------

0.4.1 (2013-10-08)
------------------

0.4.0 (2013-08-29)
------------------
* Add bugtracker and repo info URLs.
* Changelogs at package level.
* Separate and comment velocity feedback remaps.
* License link fixed.

0.3.0 (2013-07-02)
------------------
* Fix on velocity smoother to deal with low-rate simulated time (namely Stage).
* Allow using end velocity commands as robot feedback (until now we can use only odometry).

0.2.3 (2013-04-15)
------------------

0.2.2 (2013-02-10)
------------------

0.2.1 (2013-02-08)
------------------

0.2.0 (2013-02-07)
------------------
* Catkinized.

0.1.3 (2013-01-08)
------------------
* Dynamic reconfigure for velocity/acceleration limits.
* Fix on deceleration smoothing.

0.1.2 (2013-01-02)
------------------
* Add test program.
* Add licensing.

0.1.1 (2012-12-21)
------------------
* Keep direction constant when smoothing velocities, i.e. draw constant arcs. To do so we must sometimes over-limit dv or dw. 
* Bound velocity in addition to acceleration. Also set physically meaningful values for acceleration.

0.1.0 (2012-12-05)
------------------
* Initial version.
