"""Smoke test: confirm DirectML works with onnxruntime + GPU detection."""
import onnxruntime as ort

print("ONNX Runtime version:", ort.__version__)
print("\nAvailable providers:")
for p in ort.get_available_providers():
    print(f"  {p}")

# Test DirectML provider
print("\nDirectML test:")
try:
    sess_options = ort.SessionOptions()
    # Just initialise the providers — that confirms the GPU is accessible
    providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    print(f"  Requested providers: {providers}")
    print(f"  DmlExecutionProvider available: {'DmlExecutionProvider' in ort.get_available_providers()}")
except Exception as e:
    print(f"  FAILED: {e}")

# Test fastembed with DirectML
print("\nFastEmbed GPU test:")
try:
    import time
    from fastembed import TextEmbedding
    print("  Loading model with DirectML provider...")
    t0 = time.perf_counter()
    model = TextEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.2f}s")

    # Embed a batch of test sentences
    test_sentences = [
        "This agreement shall be governed by the laws of Delaware.",
        "Either party may terminate with 30 days written notice.",
        "Service credits not to exceed 30% of monthly fees.",
    ] * 20  # 60 sentences for a meaningful benchmark

    t0 = time.perf_counter()
    embeddings = list(model.embed(test_sentences))
    embed_time = time.perf_counter() - t0
    print(f"  Embedded {len(test_sentences)} sentences in {embed_time:.2f}s")
    print(f"  Speed: {len(test_sentences)/embed_time:.1f} sentences/sec")
    print(f"  Vector dim: {len(embeddings[0])}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

# Compare with CPU-only fastembed
print("\nFastEmbed CPU baseline:")
try:
    import time
    from fastembed import TextEmbedding
    t0 = time.perf_counter()
    model_cpu = TextEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        providers=["CPUExecutionProvider"],
    )
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.2f}s")

    test_sentences = [
        "This agreement shall be governed by the laws of Delaware.",
        "Either party may terminate with 30 days written notice.",
        "Service credits not to exceed 30% of monthly fees.",
    ] * 20

    t0 = time.perf_counter()
    _ = list(model_cpu.embed(test_sentences))
    embed_time = time.perf_counter() - t0
    print(f"  Embedded {len(test_sentences)} sentences in {embed_time:.2f}s")
    print(f"  Speed: {len(test_sentences)/embed_time:.1f} sentences/sec")
except Exception as e:
    print(f"  FAILED: {e}")
