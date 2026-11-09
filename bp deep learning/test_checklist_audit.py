import json
import math
import numpy as np
from pathlib import Path

def test_audit():
    json_path = Path("scratch/gemastik/Bp revisi/checklist_results.json")
    assert json_path.exists(), "checklist_results.json not found"
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print("=== AUDITING 15-POINT CHECKLIST SCHEMA & VALUES ===")
    
    # 0. Metadata & Summary
    meta = data["metadata"]
    assert meta["task_id"] == "TASK-CHECKLIST-15-EVAL"
    assert "execution_time_seconds" in meta
    print(f"Metadata verified. Execution time: {meta['execution_time_seconds']}s")
    
    dsum = data["dataset_summary"]
    assert dsum["raw_segments"] == 732
    assert dsum["raw_subjects"] == 75
    assert dsum["tight_sqi_segments"] == 472
    assert dsum["tight_sqi_subjects"] == 57
    assert dsum["disjoint_cal_eval_segments"] == 415
    assert dsum["anchor_excluded_segments"] == 41
    print("Dataset summary verified.")
    
    # 1. Baseline comparison
    p1 = data["1_baseline_comparison"]
    for m in ["zero_shot", "naive_tl", "modified_tl_v2"]:
        assert m in p1
        for cond in ["uncalibrated", "calibrated"]:
            assert cond in p1[m]
            for target in ["sbp", "dbp"]:
                met = p1[m][cond][target]
                assert met["mae"] <= met["rmse"], f"MAE > RMSE in {m} {cond} {target}"
                # Check RMSE ≈ sqrt(bias^2 + std^2)
                expected_rmse = math.sqrt(met["signed_mean_bias"]**2 + met["error_std"]**2)
                diff = abs(met["rmse"] - expected_rmse)
                assert diff < 0.05, f"RMSE mismatch in {m} {cond} {target}: got {met['rmse']}, expected {expected_rmse:.2f}"
    print("Point 1 (Baseline comparison): PASS")
    
    # 2. Ablation study
    p2 = data["2_ablation_study"]
    expected_abl = [
        "full_modified_tl_v2",
        "ablation_a_without_l2sp",
        "ablation_b_without_cal_loss",
        "ablation_c_pure_linear_probing"
    ]
    for k in expected_abl:
        assert k in p2
        assert "loss" in p2[k]
        assert "sbp_cal" in p2[k]
        assert "dbp_cal" in p2[k]
    print("Point 2 (Ablation study): PASS")
    
    # 3. Calibration impact
    p3 = data["3_calibration_impact"]
    for m in ["zero_shot", "naive_tl", "modified_tl_v2", "ablation_a_no_l2sp", "ablation_b_no_cal_loss", "ablation_c_pure_linear_probe"]:
        assert m in p3
        assert p3[m]["disjoint_n_eval"] == 415
        sbp = p3[m]["sbp"]
        dbp = p3[m]["dbp"]
        # reduction = uncal - cal
        assert round(sbp["uncal_mae"] - sbp["cal_mae"], 2) == sbp["mae_reduction_mmHg"]
        assert round(dbp["uncal_mae"] - dbp["cal_mae"], 2) == dbp["mae_reduction_mmHg"]
    print("Point 3 (Calibration impact side-by-side): PASS")
    
    # 4. CV robustness
    p4 = data["4_cross_validation_robustness"]
    assert p4["n_splits"] == 3
    assert len(p4["folds"]) == 3
    for f in p4["folds"]:
        assert f["n_test_samples"] > 0
        assert f["n_test_subjects"] > 0
    print("Point 4 (Cross-validation robustness): PASS")
    
    # 5. Generalization
    p5 = data["5_generalization"]
    assert "in_fold_train_error" in p5
    assert "out_of_fold_test_error_uncal" in p5
    assert "out_of_fold_test_error_cal" in p5
    gap_s = round(p5["out_of_fold_test_error_uncal"]["sbp_mae"] - p5["in_fold_train_error"]["sbp_mae"], 2)
    assert abs(p5["generalization_gap_uncal_sbp"] - gap_s) < 0.01
    print("Point 5 (Generalization): PASS")
    
    # 6. Regression metrics
    p6 = data["6_regression_metrics"]["modified_tl_v2_calibrated"]
    assert p6["sbp"]["mae"] > 0 and p6["dbp"]["mae"] > 0
    print("Point 6 (Regression metrics): PASS")
    
    # 7. BHS error thresholds
    p7 = data["7_bhs_error_thresholds"]["modified_tl_v2_calibrated"]
    for tgt in ["sbp", "dbp"]:
        t = p7[tgt]
        assert t["pct_le_5"] <= t["pct_le_10"] <= t["pct_le_15"]
        # check grade
        if t["pct_le_5"] >= 60 and t["pct_le_10"] >= 85 and t["pct_le_15"] >= 95:
            expected_g = "A"
        elif t["pct_le_5"] >= 50 and t["pct_le_10"] >= 75 and t["pct_le_15"] >= 90:
            expected_g = "B"
        elif t["pct_le_5"] >= 40 and t["pct_le_10"] >= 65 and t["pct_le_15"] >= 85:
            expected_g = "C"
        else:
            expected_g = "D"
        assert t["bhs_grade"] == expected_g
    print("Point 7 (BHS error thresholds & grading): PASS")
    
    # 8. AAMI criteria
    p8 = data["8_aami_criteria"]["modified_tl_v2_calibrated"]
    assert p8["sbp_aami_status"] in ["PASS", "FAIL"]
    assert p8["dbp_aami_status"] in ["PASS", "FAIL"]
    assert p8["overall_aami_verdict"] == ("PASS" if p8["sbp_aami_status"] == "PASS" and p8["dbp_aami_status"] == "PASS" else "FAIL")
    print("Point 8 (AAMI criteria): PASS")
    
    # 9. Per-subject analysis
    p9 = data["9_per_subject_analysis"]
    assert p9["n_subjects_evaluated"] == 41  # 41 multi-segment subjects
    assert len(p9["top_5_best_subjects"]) == 5
    assert len(p9["top_5_worst_outliers"]) == 5
    assert "cohort_distribution" in p9
    print("Point 9 (Per-subject error analysis): PASS")
    
    # 10. Calibration robustness
    p10 = data["10_calibration_robustness"]
    assert "anchor_segment_0_first" in p10
    assert "anchor_middle_segment" in p10
    assert "anchor_max_cardiac_psd" in p10
    for strat in p10:
        assert p10[strat]["n_eval_segments"] == 415
    print("Point 10 (Calibration anchor robustness): PASS")
    
    # 11. SQI sensitivity
    p11 = data["11_sqi_sensitivity"]
    assert p11["tight_sqi_filtered"]["n_segments"] == 472
    assert p11["full_unfiltered"]["n_segments"] == 732
    print("Point 11 (SQI sensitivity): PASS")
    
    # 12. Label limitation
    p12 = data["12_label_limitation"]
    assert p12["ground_truth_cuff_label"]["sbp_within_subject_std_mean"] == 0.0
    assert p12["model_predictions_dynamic_variance"]["pred_sbp_within_subject_std_mean"] > 0.0
    print("Point 12 (Label limitation analysis): PASS")
    
    # 13. Training stability
    p13 = data["13_training_stability"]
    assert p13["seeds_evaluated"] == [42, 123, 999]
    assert len(p13["runs"]) == 3
    print("Point 13 (Training stability across seeds): PASS")
    
    # 14. Data leakage audit
    p14 = data["14_data_leakage_audit"]
    assert p14["train_test_subject_separation"]["status"] == "PASS"
    assert p14["train_only_target_zscoring"]["status"] == "PASS"
    assert p14["calibration_anchor_exclusion"]["status"] == "PASS"
    assert p14["calibration_anchor_exclusion"]["anchor_leakage_detected"] is False
    print("Point 14 (Data leakage programmatic audit): PASS")
    
    # 15. Computational efficiency
    p15 = data["15_computational_efficiency"]
    assert p15["total_script_elapsed_seconds"] < 20.0
    assert p15["single_segment_inference_latency_ms"] < 100.0
    assert p15["parameter_counts"]["pct_parameters_fine_tuned"] < 15.0
    print("Point 15 (Computational efficiency): PASS")
    
    print("\nALL 15 POINTS PASSED STRICT SCHEMA & CONSISTENCY AUDIT!")

if __name__ == "__main__":
    test_audit()
