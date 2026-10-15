"""Print the vLLM 0.11 AWQ linear-method source that the integration has to replace (development aid)."""
import inspect

from vllm.model_executor.layers.quantization import awq, awq_marlin

print(inspect.getsource(awq.AWQLinearMethod))
print("=" * 80)
src = inspect.getsource(awq_marlin.AWQMarlinLinearMethod)
print(src[:2500])
