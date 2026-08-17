# Pseudo-label policy leaderboard

| domain | policy | frames | boxes | mean conf | Empty | Tweezers | Needle | rejections |
|---|---|---|---|---|---|---|---|---|
| ID | sup1280_id_p1 | 689/720 | 1636 | 0.855 | 422 | 653 | 561 | {"frame_conf_threshold": 12} |
| ID | sup_ref640_id_p1 | 702/720 | 1690 | 0.876 | 470 | 651 | 569 | {"frame_conf_threshold": 11} |
| ID | sup_ref640_id_p2 | 691/720 | 1569 | 0.894 | 393 | 639 | 537 | {"frame_conf_threshold": 13} |
| ID | sup_ref640_id_p3 | 574/720 | 1282 | 0.907 | 323 | 525 | 434 | {"frame_conf_threshold": 195} |
| ID | sup_ref640_id_p3_sanity_temporal | 566/720 | 1020 | 0.912 | 249 | 471 | 300 | {"frame_conf_threshold": 167, "max_detections_per_frame": 207, "min_box_height": 1, "temporal_min_hits": 82} |
| ID | sup_ref640_id_p4 | 69/720 | 168 | 0.938 | 97 | 52 | 19 | {"frame_conf_threshold": 720} |
| OOD | ssl_id_temporal_ood_conf | 0/380 | 0 |  | 0 | 0 | 0 | {"frame_conf_threshold": 194} |
| OOD | ssl_id_temporal_ood_conf_sanity | 0/380 | 0 |  | 0 | 0 | 0 | {"frame_conf_threshold": 178, "max_box_area": 16} |
| OOD | ssl_id_temporal_ood_conf_temporal | 0/380 | 0 |  | 0 | 0 | 0 | {"frame_conf_threshold": 178, "temporal_min_hits": 16} |
| OOD | ssl_id_temporal_ood_conservative | 0/380 | 0 |  | 0 | 0 | 0 | {"class_or_box_conf_threshold": 188, "temporal_min_hits": 6} |
| OOD | ssl_id_temporal_ood_relaxed_combined | 123/380 | 129 | 0.891 | 0 | 9 | 120 | {"class_cap_2": 43, "class_or_box_conf_threshold": 102, "frame_conf_threshold": 22, "max_box_area": 16, "temporal_min_hits": 21} |
| OOD | ssl_id_temporal_ood_relaxed_conf | 193/380 | 235 | 0.867 | 1 | 19 | 215 | {"frame_conf_threshold": 98} |
| OOD | ssl_id_temporal_ood_relaxed_sanity | 177/380 | 219 | 0.866 | 1 | 19 | 199 | {"frame_conf_threshold": 84, "max_box_area": 30} |
| OOD | ssl_id_temporal_ood_relaxed_temporal | 189/380 | 219 | 0.871 | 0 | 15 | 204 | {"frame_conf_threshold": 81, "temporal_min_hits": 33} |
