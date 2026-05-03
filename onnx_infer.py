import onnxruntime as ort
import numpy as np
import logging

class ONNXModel:
    def __init__(self, model_path):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Use CPU-only execution (no GPU available locally)
        self.logger.info("Initializing ONNX Runtime with CPU provider...")
        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"]
            )
            self.logger.info("Successfully initialized with CPU provider.")
        except Exception as e:
            self.logger.error(f"Failed to initialize ONNX Runtime: {e}")
            raise

        self.input_name = self.session.get_inputs()[0].name


    def infer(self, input_data):
        outputs = self.session.run(
            None,
            {self.input_name: input_data}
        )
        return outputs[0]
