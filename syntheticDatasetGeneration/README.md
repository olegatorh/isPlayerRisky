Dataset design

•	One row represents an aggregated player profile over a 30-day observation window.

•	Features simulate account, financial, session, and behavior-related gambling activity.

•	The target is_risky_player is generated from a latent probabilistic risk function driven by behavioral risk indicators such as high night activity, elevated losses, repeated failed deposits, long session duration, and chasing-like behavior.

•	Train and test splits intentionally include mild distribution drift to emulate production conditions.
