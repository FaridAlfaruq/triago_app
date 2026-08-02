import os
import unittest
from unittest.mock import patch

import serial

from akuisisi_data.get_stm32 import find_stm32_port, stream_stm32_data
from GUI.plot_page import STM32Worker


class SensorErrorFlowTests(unittest.TestCase):
    def test_environment_port_has_priority(self):
        with patch.dict(os.environ, {"TRIAGO_SERIAL_PORT": "COM42"}):
            self.assertEqual(find_stm32_port(), "COM42")

    def test_missing_sensor_does_not_guess_hardcoded_port(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("akuisisi_data.get_stm32.list_ports.comports", return_value=[]),
        ):
            self.assertIsNone(find_stm32_port())

    def test_missing_sensor_raises_actionable_error(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("akuisisi_data.get_stm32.list_ports.comports", return_value=[]),
        ):
            stream = stream_stm32_data()
            with self.assertRaisesRegex(serial.SerialException, "Port STM32 tidak ditemukan"):
                next(stream)

    def test_connected_but_silent_sensor_times_out(self):
        class SilentSerial:
            in_waiting = 0
            is_open = True
            dtr = False
            rts = False

            def set_buffer_size(self, **_kwargs):
                pass

            def reset_input_buffer(self):
                pass

            def reset_output_buffer(self):
                pass

            def write(self, _data):
                pass

            def flush(self):
                pass

            def close(self):
                self.is_open = False

        with (
            patch("akuisisi_data.get_stm32.serial.Serial", return_value=SilentSerial()),
            patch("akuisisi_data.get_stm32.time.sleep"),
            patch("akuisisi_data.get_stm32.time.monotonic", side_effect=[0.0, 5.1]),
        ):
            stream = stream_stm32_data(port="COM42", data_timeout=5.0)
            with self.assertRaisesRegex(serial.SerialTimeoutException, "Tidak ada respons serial"):
                next(stream)

    def test_worker_forwards_connection_error_to_gui_signal(self):
        errors = []
        worker = STM32Worker()
        worker.error_occurred.connect(errors.append)

        with patch(
            "GUI.plot_page.stream_stm32_data",
            side_effect=serial.SerialException("sensor disconnected"),
        ):
            worker.run()

        self.assertEqual(errors, ["sensor disconnected"])


if __name__ == "__main__":
    unittest.main()
