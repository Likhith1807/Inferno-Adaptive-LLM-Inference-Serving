"""
Reference registry with automatic numbering in order of first citation.

VERIFIED = checked against the publisher / proceedings / arXiv page during preparation of this report (marked with a
dagger in the list). The others are cited from the author's knowledge; confirm venue, year and page numbers before
any formal submission.
"""
from __future__ import annotations

from .style import S
from reportlab.platypus import Paragraph

VERIFIED = {"vllm", "orca", "sarathi", "distserve", "splitwise", "llumnix", "clipper", "alpaserve", "flash", "gptq", "awq",
            "marlin", "roofline", "openclosed", "hoefler", "georges"}

R = {
    "vaswani": "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., Polosukhin, I. "
               "<i>Attention Is All You Need.</i> NeurIPS 2017. arXiv:1706.03762.",
    "orca": "Yu, G.-I., Jeong, J. S., Kim, G.-W., Kim, S., Chun, B.-G. <i>Orca: A Distributed Serving System for "
            "Transformer-Based Generative Models.</i> USENIX OSDI 2022.",
    "vllm": "Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., Stoica, I. "
            "<i>Efficient Memory Management for Large Language Model Serving with PagedAttention.</i> ACM SOSP 2023.",
    "sarathi": "Agrawal, A., Kedia, N., Panwar, A., Mohan, J., Kwatra, N., Gulavani, B. S., Tumanov, A., Ramjee, R. "
               "<i>Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.</i> USENIX OSDI 2024, pp. 117-134.",
    "distserve": "Zhong, Y., Liu, S., Chen, J., Hu, J., Zhu, Y., Liu, X., Jin, X., Zhang, H. <i>DistServe: Disaggregating "
                 "Prefill and Decoding for Goodput-optimized Large Language Model Serving.</i> USENIX OSDI 2024.",
    "splitwise": "Patel, P., Choukse, E., Zhang, C., Shah, A., Goiri, I., Maleki, S., Bianchini, R. <i>Splitwise: Efficient "
                 "Generative LLM Inference Using Phase Splitting.</i> ACM/IEEE ISCA 2024. arXiv:2311.18677.",
    "llumnix": "Sun, B., Huang, Z., Zhao, H., Xiao, W., Zhang, X., Li, Y., Lin, W. <i>Llumnix: Dynamic Scheduling for Large "
               "Language Model Serving.</i> USENIX OSDI 2024.",
    "clipper": "Crankshaw, D., Wang, X., Zhou, G., Franklin, M. J., Gonzalez, J. E., Stoica, I. <i>Clipper: A Low-Latency "
               "Online Prediction Serving System.</i> USENIX NSDI 2017, pp. 613-627.",
    "alpaserve": "Li, Z., Zheng, L., Zhong, Y., Liu, V., Sheng, Y., Jin, X., Huang, Y., Chen, Z., Zhang, H., Gonzalez, J. E., "
                 "Stoica, I. <i>AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving.</i> "
                 "USENIX OSDI 2023, pp. 663-679.",
    "infaas": "Romero, F., Li, Q., Yadwadkar, N. J., Kozyrakis, C. <i>INFaaS: Automated Model-less Inference Serving.</i> "
              "USENIX ATC 2021. arXiv:1905.13348.",
    "flash": "Dao, T., Fu, D. Y., Ermon, S., Rudra, A., Re, C. <i>FlashAttention: Fast and Memory-Efficient Exact Attention "
             "with IO-Awareness.</i> NeurIPS 2022. arXiv:2205.14135.",
    "flash2": "Dao, T. <i>FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.</i> ICLR 2024. "
              "arXiv:2307.08691.",
    "gptq": "Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. <i>GPTQ: Accurate Post-Training Quantization for "
            "Generative Pre-trained Transformers.</i> ICLR 2023. arXiv:2210.17323.",
    "awq": "Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.-M., Wang, W.-C., Xiao, G., Dang, X., Gan, C., Han, S. "
           "<i>AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.</i> MLSys 2024 (Best Paper).",
    "llmint8": "Dettmers, T., Lewis, M., Belkada, Y., Zettlemoyer, L. <i>LLM.int8(): 8-bit Matrix Multiplication for "
               "Transformers at Scale.</i> NeurIPS 2022. arXiv:2208.07339.",
    "smoothquant": "Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., Han, S. <i>SmoothQuant: Accurate and Efficient "
                   "Post-Training Quantization for Large Language Models.</i> ICML 2023. arXiv:2211.10438.",
    "marlin": "Frantar, E., Castro, R. L., Chen, J., Hoefler, T., Alistarh, D. <i>MARLIN: Mixed-Precision Auto-Regressive "
              "Parallel Inference on Large Language Models.</i> ACM PPoPP 2025. arXiv:2408.11743.",
    "pope": "Pope, R., Douglas, S., Chowdhery, A., Devlin, J., Bradbury, J., Levskaya, A., Heek, J., Xiao, K., Agrawal, S., "
            "Dean, J. <i>Efficiently Scaling Transformer Inference.</i> MLSys 2023. arXiv:2211.05102.",
    "spec": "Leviathan, Y., Kalman, M., Matias, Y. <i>Fast Inference from Transformers via Speculative Decoding.</i> "
            "ICML 2023. arXiv:2211.17192.",
    "sglang": "Zheng, L., Yin, L., Xie, Z., Sun, C., Huang, J., Yu, C. H., Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., "
              "Barrett, C., Sheng, Y. <i>SGLang: Efficient Execution of Structured Language Model Programs.</i> NeurIPS 2024. "
              "arXiv:2312.07104.",
    "roofline": "Williams, S., Waterman, A., Patterson, D. <i>Roofline: An Insightful Visual Performance Model for Multicore "
                "Architectures.</i> Communications of the ACM 52(4):65-76, 2009.",
    "logp": "Culler, D., Karp, R., Patterson, D., Sahay, A., Schauser, K. E., Santos, E., Subramonian, R., von Eicken, T. "
            "<i>LogP: Towards a Realistic Model of Parallel Computation.</i> ACM PPoPP 1993.",
    "openmpi": "Gabriel, E., Fagg, G. E., Bosilca, G., Angskun, T., Dongarra, J. J., Squyres, J. M., et al. <i>Open MPI: Goals, "
               "Concept, and Design of a Next Generation MPI Implementation.</i> EuroPVM/MPI 2004.",
    "mpi": "Message Passing Interface Forum. <i>MPI: A Message-Passing Interface Standard, Version 4.0.</i> 2021.",
    "openclosed": "Schroeder, B., Wierman, A., Harchol-Balter, M. <i>Open Versus Closed: A Cautionary Tale.</i> USENIX NSDI 2006.",
    "tail": "Dean, J., Barroso, L. A. <i>The Tail at Scale.</i> Communications of the ACM 56(2):74-80, 2013.",
    "georges": "Georges, A., Buytaert, D., Eeckhout, L. <i>Statistically Rigorous Java Performance Evaluation.</i> ACM OOPSLA "
               "2007, pp. 57-76.",
    "hoefler": "Hoefler, T., Belli, R. <i>Scientific Benchmarking of Parallel Computing Systems: Twelve Ways to Tell the "
               "Masses When Reporting Performance Results.</i> SC 2015.",
    "mytkowicz": "Mytkowicz, T., Diwan, A., Hauswirth, M., Sweeney, P. F. <i>Producing Wrong Data Without Doing Anything "
                 "Obviously Wrong!</i> ACM ASPLOS 2009.",
    "kalibera": "Kalibera, T., Jones, R. <i>Rigorous Benchmarking in Reasonable Time.</i> ACM ISMM 2013.",
    "mlperf": "Reddi, V. J., Cheng, C., Kanter, D., Mattson, P., et al. <i>MLPerf Inference Benchmark.</i> ACM/IEEE ISCA 2020.",
    "little": "Little, J. D. C. <i>A Proof for the Queuing Formula: L = &lambda;W.</i> Operations Research 9(3):383-387, 1961.",
    "wilcoxon": "Wilcoxon, F. <i>Individual Comparisons by Ranking Methods.</i> Biometrics Bulletin 1(6):80-83, 1945.",
    "holm": "Holm, S. <i>A Simple Sequentially Rejective Multiple Test Procedure.</i> Scandinavian Journal of Statistics "
            "6(2):65-70, 1979.",
    "cohen": "Cohen, J. <i>Statistical Power Analysis for the Behavioral Sciences</i>, 2nd ed. Lawrence Erlbaum, 1988.",
    "efron": "Efron, B. <i>Bootstrap Methods: Another Look at the Jackknife.</i> Annals of Statistics 7(1):1-26, 1979.",
    "student": "Student (W. S. Gosset). <i>The Probable Error of a Mean.</i> Biometrika 6(1):1-25, 1908.",
    "qwen": "Qwen Team. <i>Qwen2.5 Technical Report.</i> arXiv:2412.15115, 2024.",
    "cudaguide": "NVIDIA. <i>CUDA C++ Programming Guide</i> and <i>CUDA C++ Best Practices Guide</i> (shared memory banks, memory "
                 "coalescing, warp-level primitives). NVIDIA documentation.",
    "nsight": "NVIDIA. <i>Nsight Compute Documentation</i> and <i>NVIDIA GPU Performance Counter Permissions (ERR_NVGPUCTRPERM).</i> "
              "NVIDIA developer documentation.",
    "mps": "NVIDIA. <i>Multi-Process Service (MPS)</i> and <i>Multi-Instance GPU (MIG)</i> user guides. NVIDIA documentation.",
    "vllmdocs": "vLLM project. <i>vLLM documentation and source</i> (v0.11.0): OpenAI-compatible server, quantization backends "
                "(awq, awq_marlin, gptq_marlin), scheduler parameters (max_num_seqs, max_num_batched_tokens). "
                "https://github.com/vllm-project/vllm",
}

_order: list[str] = []


def cite(*keys: str) -> str:
    nums = []
    for k in keys:
        if k not in R:
            raise KeyError(k)
        if k not in _order:
            _order.append(k)
        nums.append(_order.index(k) + 1)
    return "[" + ",".join(str(n) for n in sorted(nums)) + "]"


def reference_flowables():
    out = []
    for i, k in enumerate(_order, 1):
        dag = " †" if k in VERIFIED else ""
        out.append(Paragraph(f"[{i}]&nbsp;&nbsp;{R[k]}{dag}", S["ref"]))
    return out
