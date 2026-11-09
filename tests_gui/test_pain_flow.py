"""Headless GUI integration tests; run in the kiosk/GUI Python environment."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
from PyQt6.QtWidgets import QApplication
from GUI.regist_page import RegistrationPage
from GUI.output_page import OutputPage
from GUI.loading_page import ProcessingWorker
from GUI.main_gui import TriaGoApplication
from model.deployment_inference import TriageOnnxModel


def result(pain=8):
    return {"bed": "A1", "gcs": 15, "pain_score": pain, "hr": 75, "rr": 16,
            "spo2": 98, "temperature": 36.5, "systolic": 120, "diastolic": 80,
            "triage_status": "DARURAT", "triage_valid": True, "xgboost_score": 0.91}


class PainGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.client = Mock()
        patcher = patch("GUI.output_page.TriageApiClient", return_value=self.client)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_registration_requires_explicit_pain_and_accepts_zero(self):
        page = RegistrationPage()
        self.addCleanup(page.close)
        page.handle_bed_selection(next(iter(page.bed_buttons)))
        page.handle_gcs_selection(15)
        self.assertFalse(page.btn_start.isEnabled())
        page.handle_pain_selection(0)
        self.assertTrue(page.btn_start.isEnabled())
        self.assertTrue(page.pain_buttons[0].isChecked())
        page.handle_pain_selection(8)
        self.assertFalse(page.pain_buttons[0].isChecked())
        self.assertTrue(page.pain_buttons[8].isChecked())
        page.resize(1280, 800)
        page.show()
        self.app.processEvents()
        destination = Path("runtime/qa")
        destination.mkdir(parents=True, exist_ok=True)
        page.grab().save(str(destination / "registration-pain.png"))

    def test_output_forwards_pain_and_xgboost_score_and_skips_invalid_inference(self):
        page = OutputPage()
        self.addCleanup(page.close)
        page.update_results(result(0))
        args = self.client.send_triage_result.call_args.kwargs
        self.assertEqual(args["pain_score"], 0)
        self.assertEqual(args["score"], 0.91)
        self.assertEqual(args["classification"].lower(), "yellow")
        self.assertEqual(page.lbl_pain_val.text(), "0 / 10")
        self.assertEqual(json.loads(page.iot_json_payload)["PainScore"], 0)
        page.update_results({**result(), "triage_valid": False})
        self.client.send_triage_result.assert_called_once()
        page.resize(1280, 800)
        page.show()
        self.app.processEvents()
        destination = Path("runtime/qa")
        destination.mkdir(parents=True, exist_ok=True)
        page.grab().save(str(destination / "output-pain.png"))

    def test_main_gui_metadata_and_reset_include_pain(self):
        with patch("GUI.loading_page.LoadingPage._load_triage_model"):
            window = TriaGoApplication()
        self.addCleanup(window.close)
        window.page_registration.handle_pain_selection(8)
        with tempfile.TemporaryDirectory() as directory:
            with patch("GUI.main_gui.project_root", directory):
                window.save_consolidated_csv(result(8))
            metadata = next((Path(directory) / "data_pengukuran").glob("*.json"))
            self.assertEqual(json.loads(metadata.read_text())["Pain Score"], 8)
        window.reset_to_gatekeeper()
        self.assertIsNone(window.page_registration.selected_pain)
        self.assertTrue(all(not button.isChecked() for button in window.page_registration.pain_buttons.values()))
        self.assertFalse(window.page_registration.btn_start.isEnabled())

    def test_worker_passes_registration_pain_to_real_model_and_final_result(self):
        values = np.ones(1250)
        ecg = Mock()
        ecg.downsample.return_value = (values, np.arange(1250) / 125)
        for name in ("notch", "detrending", "lowpass", "savgol"):
            getattr(ecg, name).side_effect = lambda data, **kwargs: data
        ecg.detect_r_peaks.return_value = (np.arange(100, 1250, 100), [])
        ecg.calculate_heart_rate.return_value = 75
        ecg.calculate_respiration_rate.return_value = (16, values, [])
        ecg.last_respiration_details = {"quality": 1}
        ppg = Mock()
        ppg.process_ppg.return_value = {"spo2": 98, "pi_red": 1, "pi_ir": 1,
                                       "red_clean": values, "ir_clean": values, "ppg_hr": 75}
        bp = Mock()
        bp.predict_recording.return_value = {"sqa_passed": True, "passed_segments": 1,
                                             "total_segments": 1, "sbp": 120, "dbp": 80}
        predictor = Mock(wraps=TriageOnnxModel())
        with patch("GUI.loading_page.ECGProcessor", return_value=ecg), patch("GUI.loading_page.PPGProcessor", return_value=ppg):
            worker = ProcessingWorker(values, np.arange(1250), values, values,
                                      patient_info={"bed": "A1", "gcs": 15, "pain_score": 8,
                                                    "temp_skin": 34.5, "temp_ambient": 28},
                                      triage_predictor=predictor, bpnet_predictor=bp)
        results = []
        worker.processing_finished.connect(results.append)
        with patch.object(ProcessingWorker, "msleep"):
            worker.run()
        self.assertEqual(predictor.predict.call_args.args[0]["pain_score"], 8)
        self.assertEqual(results[0]["pain_score"], 8)
        self.assertTrue(results[0]["triage_valid"])
        page = OutputPage()
        self.addCleanup(page.close)
        page.update_results(results[0])
        sent = self.client.send_triage_result.call_args.kwargs
        self.assertEqual(sent["pain_score"], 8)
        self.assertEqual(sent["score"], results[0]["xgboost_score"])


if __name__ == "__main__":
    unittest.main()
