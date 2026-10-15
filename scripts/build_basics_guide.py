"""
Builds docs/Inferno_Plain_English_Guide.pdf: a short, plain-language "understand everything before you present"
crash sheet. Not for submission -- a personal study aid, so it is written in ordinary language with analogies
first and jargon explained immediately after. Numbers match the deck and report (same source: docs/results via
report_pdf.data.facts()).

    python scripts/build_basics_guide.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer  # noqa: E402

from report_pdf import data  # noqa: E402

OUT = ROOT / "docs" / "Inferno_Plain_English_Guide.pdf"
FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))

INK = colors.HexColor("#17191c")
GREY = colors.HexColor("#55585f")
ACCENT = colors.HexColor("#a50015")
ACCENT2 = colors.HexColor("#0b7a4b")
BAND = colors.HexColor("#f5f1e8")

TITLE = ParagraphStyle("title", fontName="DV-B", fontSize=21, leading=26, textColor=ACCENT, spaceAfter=4)
SUBTITLE = ParagraphStyle("sub", fontName="DV-I", fontSize=11.5, leading=15, textColor=GREY, spaceAfter=14)
H1 = ParagraphStyle("h1", fontName="DV-B", fontSize=14.5, leading=18, textColor=ACCENT, spaceBefore=16, spaceAfter=7)
LEAD = ParagraphStyle("lead", fontName="DV-B", fontSize=11, leading=15.5, textColor=INK, spaceAfter=6)
BODY = ParagraphStyle("body", fontName="DV", fontSize=10.3, leading=14.5, textColor=INK, spaceAfter=6, alignment=TA_LEFT)
TERM = ParagraphStyle("term", fontName="DV", fontSize=10.3, leading=14.5, textColor=INK, spaceAfter=7)
BUL = ParagraphStyle("bul", fontName="DV", fontSize=10.3, leading=14.3, textColor=INK, leftIndent=12)
QA_Q = ParagraphStyle("qaq", fontName="DV-B", fontSize=10.3, leading=14, textColor=ACCENT2, spaceBefore=5, spaceAfter=1)
QA_A = ParagraphStyle("qaa", fontName="DV", fontSize=10.3, leading=14.3, textColor=INK, spaceAfter=2)
NOTE = ParagraphStyle("note", fontName="DV-I", fontSize=9, leading=12.5, textColor=GREY)


def P(t, style=BODY):
    return Paragraph(t, style)


def BULLETS(items, style=BUL):
    return ListFlowable([ListItem(Paragraph(i, style), leftIndent=14, bulletColor=ACCENT) for i in items],
                        bulletType="bullet", start="•", leftIndent=6, spaceBefore=2, spaceAfter=8)


def term(name, text):
    return P(f'<font name="DV-B" color="#a50015">{name}</font> &mdash; {text}', TERM)


def qa(q, a):
    return [P(q, QA_Q), P(a, QA_A)]


def build():
    F = data.facts()
    k = F["kern"]
    e = F["e2e"]
    r1 = F["e2e_ratio_b8"]["1"]["mean"]
    lo, hi = min(r["pct_change"] for r in F["h1"]), max(r["pct_change"] for r in F["h1"])

    S = []
    S.append(P("Inferno &mdash; Plain-English Guide", TITLE))
    S.append(P("Read this once before you present. It explains, in ordinary words, what the project is, what "
               "every term on the slides means, and why we made each choice. Every number here matches the "
               "slides and the report exactly.", SUBTITLE))

    # ---------------------------------------------------------------- 1
    S.append(P("1. The one-minute summary", H1))
    S.append(P(
        "When an AI model writes a reply, it produces the answer one word (technically, one <i>token</i>) at a "
        "time. To produce even a single word, it must read its entire &ldquo;brain&rdquo; &mdash; billions of "
        "numbers called <i>weights</i> &mdash; from the GPU's memory. That reading is the slow part, not the "
        "thinking. Our project asks one question: <b>can we make this faster</b> by (a) hand-writing a smarter "
        "GPU program that reads less data, and (b) cleverly choosing which running copy of the model should "
        "answer each incoming request? We built both, tested them very carefully with repeated, honest "
        "measurements, and reported plainly what worked and what did not.", BODY))

    # ---------------------------------------------------------------- 2
    S.append(P("2. What we have done, and what is left", H1))
    S.append(P("Done so far:", LEAD))
    S.append(BULLETS([
        "Wrote and tested our own fast GPU program (a <b>kernel</b>) &mdash; it works and runs close to the fastest speed physically possible on our GPU.",
        "Plugged it into a real, production AI-serving system (<b>vLLM</b>) and measured real speed with real generated text.",
        "Built a &ldquo;smart router&rdquo; idea (<b>adaptive routing</b>) and fairly tested it against simple fixed choices, over 96 separate timed experiments.",
        "Built a message-passing router (<b>MPI</b>) so several running copies of the model can be coordinated.",
        "Built a live, interactive dashboard and a full written report &mdash; every number in them is generated automatically from real recorded data, never typed by hand.",
    ]))
    S.append(P("Left to do, by the end of November:", LEAD))
    S.append(BULLETS([
        "Test everything on <b>multiple, separate GPUs</b>. Right now everything runs squeezed onto ONE 8&nbsp;GB GPU, which changed our results (explained in Section 5).",
        "Use a special GPU profiling tool (<b>Nsight Compute</b>) to directly confirm, with hard counters, exactly why each step of our program got faster.",
        "Extend our kernel so it stays fast even when many users ask questions at the same time, not just one.",
    ]))

    # ---------------------------------------------------------------- 3
    S.append(P("3. The basic idea, as a story", H1))
    S.append(P(
        "Picture the AI model as a huge library. Every time it must answer, it has to walk past every single "
        "shelf &mdash; every weight/number &mdash; once, even just to produce one word. <b>This walk is why it "
        "is slow, not the thinking.</b> There are two ways to speed up the walk: (1) shrink the books &mdash; "
        "shrink the numbers themselves &mdash; so the walk covers less data; this is called "
        "<b>quantization</b>. (2) Hire more librarians &mdash; run several copies of the model &mdash; and have "
        "a receptionist decide who answers which question; this is <b>adaptive routing</b>. Our project builds "
        "and tests both ideas.", BODY))

    # ---------------------------------------------------------------- 4 glossary
    S.append(P("4. Every term on the slides, explained simply", H1))
    S.append(P(
        "This section is written so you can jump to any word you are unsure of. Read top to bottom once before "
        "presenting; use it as a lookup during the viva.", NOTE))
    S.append(Spacer(1, 4))

    S.append(term("LLM (Large Language Model)",
                  "the AI model itself &mdash; like a mini ChatGPT. It is made of billions of numbers "
                  "(<i>weights</i>) that decide what word comes next."))
    S.append(term("Token",
                  "a small piece of text, roughly a word or part of one. The model produces text one token at "
                  "a time; that is why speed is measured in &ldquo;tokens per second.&rdquo;"))
    S.append(term("GPU",
                  "a chip built to do many simple calculations (like multiplications) at the exact same time. "
                  "It is why AI is fast at all &mdash; but it has a strict speed limit on how fast it can "
                  "<i>read</i> data from its own memory, and that limit is what our project is really fighting."))
    S.append(term("Qwen2.5-1.5B",
                  "the actual AI model we used for every experiment. &ldquo;1.5B&rdquo; means 1.5 billion "
                  "numbers (parameters). We chose a model this size because our GPU has only 8&nbsp;GB of "
                  "memory &mdash; a bigger, 8-billion-parameter model simply would not fit."))
    S.append(term("FP16 (16-bit floating point)",
                  "the normal, full-precision way to store each weight (2 bytes per number). Accurate, but "
                  "takes more memory to store and more time to read."))
    S.append(term("INT4 / Quantization",
                  "instead of storing each weight at full precision, we squeeze it down to just 4 bits (about "
                  "a quarter of the space). The model becomes about 4&times; smaller, so the GPU has to read "
                  "about 4&times; less data &mdash; and since reading data is the bottleneck, this directly "
                  "makes generation faster. The cost is a small, usually acceptable, drop in numeric accuracy."))
    S.append(term("AWQ-Int4",
                  "one specific, well-known <i>recipe</i> for doing that 4-bit squeezing intelligently &mdash; "
                  "it looks at which weights matter most before shrinking them, so accuracy barely drops. We "
                  "used the official AWQ-Int4 release of our model rather than inventing our own quantization."))
    S.append(term("GPTQ",
                  "another popular 4-bit quantization recipe, similar in spirit to AWQ. We mention it in our "
                  "literature review as related work; our own experiments use AWQ-Int4 checkpoints."))
    S.append(term("CUDA",
                  "NVIDIA's programming toolkit for writing your own custom programs that run directly on the "
                  "GPU. We used CUDA to hand-write our own version of the &ldquo;read the 4-bit weights and do "
                  "the math&rdquo; step, built specifically for our compressed format."))
    S.append(term("Kernel (in GPU programming)",
                  "just means one specific small GPU program. Our &ldquo;fused int4 kernel&rdquo; is the "
                  "program we wrote that reads the compressed weights and multiplies them, all in a single, "
                  "efficient step, without ever re-expanding them back to full size in memory."))
    S.append(term("cuBLAS",
                  "NVIDIA's own ready-made, already highly optimized library for standard math like matrix "
                  "multiplication, run at full precision (FP16). We use it purely as our <b>fair-comparison "
                  "baseline</b> &mdash; &ldquo;how fast is the normal, off-the-shelf way?&rdquo; &mdash; so we "
                  "can prove our own kernel is genuinely faster, not just different."))
    S.append(term("Marlin",
                  "not something we built &mdash; it is an existing, extremely well-optimized 4-bit kernel that "
                  "ships inside vLLM, written by other researchers. It is our main real-world &ldquo;competitor&rdquo; "
                  "for judging how good our own hand-written kernel is."))
    S.append(term("Optimization ladder (v0 &rarr; v6)",
                  "we did not write our fast kernel in one attempt. We built it in seven small versions, each "
                  "fixing exactly one inefficiency in the version before it (for example: v0&rsquo;s memory "
                  "access was disorganized, so v1 fixed that; v2 introduced a hidden memory &ldquo;traffic "
                  "jam,&rdquo; so v3 removed it; and so on). Measuring every step means we can prove each idea "
                  "actually helped &mdash; and honestly report the couple of ideas that did not."))
    S.append(term("vLLM",
                  "a popular, production-quality, ready-made system (built by other people) for actually "
                  "serving an LLM to many chat users efficiently. We use it as our serving foundation and plug "
                  "our own kernel into it, so we can test our kernel inside a real, complete system rather than "
                  "in an artificial isolated test."))
    S.append(term("Why vLLM, specifically?",
                  "because it is the most widely used, well-tested, open-source serving engine available. "
                  "Comparing against it means comparing against a serious real-world system, not something we "
                  "made up ourselves. It already solves the hard parts (handling many users, managing "
                  "conversation memory), so we can focus on the one piece we are contributing: the kernel."))
    S.append(term("Parallelization &mdash; the 3 &ldquo;levels&rdquo;",
                  "&ldquo;parallel&rdquo; means doing many things at once instead of one after another. We use "
                  "three different tools at three different scales:"))
    S.append(BULLETS([
        "<b>Level 1 &mdash; CUDA, inside the GPU:</b> while computing one single answer, thousands of tiny multiplications happen at the same instant inside the chip. This is what our kernel controls.",
        "<b>Level 2 &mdash; vLLM, across users on the same GPU:</b> one GPU can work on several different users&rsquo; requests at the same time (called <i>batching</i>) instead of finishing one person fully before starting the next.",
        "<b>Level 3 &mdash; MPI, across separate replicas/machines:</b> if you run several full copies of the system (on different GPUs), you need a router to decide which copy handles each new request.",
    ]))
    S.append(term("MPI (Message Passing Interface)",
                  "a long-standing, standard toolkit from the supercomputing world that lets separate programs "
                  "&mdash; possibly on different physical machines &mdash; send short messages to each other and "
                  "coordinate. We use it to build a router: one process decides which model copy should handle "
                  "each new request and sends it a short &ldquo;handle this one&rdquo; message."))
    S.append(term("Adaptive routing / adaptive controller",
                  "our own &ldquo;smart receptionist&rdquo; idea: instead of always sending every request to the "
                  "same copy, it watches how busy each copy is and how long each request looks, and decides in "
                  "real time which copy should get the next request, aiming to keep everyone&rsquo;s wait time low."))
    S.append(term("Fixed policy",
                  "the deliberately &ldquo;boring&rdquo; comparison: always send every request to the same one "
                  "copy, with no cleverness at all. We compare our smart router against these simple, fixed "
                  "choices to check whether being clever actually pays off."))
    S.append(term("p99 latency",
                  "the wait time experienced by 99 out of 100 users, or better &mdash; i.e. how slow it is for "
                  "almost everyone, including the unlucky ones, not just the average person. Systems can look "
                  "fine on average while some real users wait a very long time; p99 catches that."))
    S.append(term("Why did the &ldquo;smart&rdquo; adaptive router lose?",
                  "because we only have <b>one physical GPU</b>, and both copies of the model live on that "
                  "same GPU. When the second copy is also made to do work, the two copies must take turns "
                  "using the same GPU hardware (&ldquo;time-slicing&rdquo;, like two teachers sharing one "
                  "whiteboard) &mdash; and that turn-taking is slower than just using one copy properly. On "
                  "genuinely separate GPUs this problem should not exist &mdash; testing that is exactly our "
                  "November multi-GPU plan."))
    S.append(term("Paired experiments / statistics / p-value",
                  "whenever we say &ldquo;X was worse than Y,&rdquo; we do not just look at one run and guess. "
                  "We repeat every experiment 8 times with the <i>exact same</i> simulated traffic for every "
                  "method being compared (a completely fair race), then run standard statistical tests to check: "
                  "is this difference real, or could it just be random luck? We only report something as a "
                  "finding once the maths says it is very unlikely to be luck."))
    S.append(term("Dashboard",
                  "a live webpage we built ourselves that shows all our results as interactive graphs &mdash; "
                  "kernel speed, the routing experiment, and more. It is proof that this is a real, working, "
                  "tested system, not just claims on slides: every number on it is generated automatically from "
                  "our raw recorded log files."))

    # ---------------------------------------------------------------- 5 results
    S.append(P("5. Our results, in plain words", H1))
    S.append(BULLETS([
        f"<b>Our kernel is fast.</b> It is about {k['speedup']:.1f}&times; faster than the normal (FP16) way of doing the same calculation, "
        f"and uses about {k['pct']:.0f}% of the absolute maximum speed the GPU's memory can physically deliver &mdash; close to the "
        f"physical limit of this hardware.",
        f"<b>Inside the real system, we tied the professional kernel.</b> Plugged into vLLM with one user talking to it, our kernel reached "
        f"{e['inferno_b8']['1']:.0f} tokens/second &mdash; about {r1:.2f}&times; Marlin's speed, i.e. a tie, not a win. Building something from scratch "
        f"that matches a professional, already-optimized library is itself a real result. With several users at once, Marlin pulls ahead, because it "
        f"uses a more advanced part of the GPU (Tensor Cores) that our kernel does not use yet.",
        f"<b>Our &ldquo;smart routing&rdquo; idea did not win &mdash; and we know exactly why.</b> Across 96 fair, repeated experiments, the adaptive "
        f"router was {lo:.0f}% to {hi:.0f}% <i>worse</i> on tail latency than simply picking one fixed setup. We traced this to the two copies fighting "
        f"over one shared GPU. This is an honest negative result, but a useful one: we know the cause, and we have a specific next experiment planned "
        f"(separate GPUs) to properly test the original idea.",
        "<b>Our MPI router itself works and is fast.</b> Sending a message between two copies takes only a couple of millionths of a second &mdash; "
        "it is not the bottleneck; the shared GPU is.",
    ]))

    # ---------------------------------------------------------------- 6 how
    S.append(P("6. How we actually ran the project, step by step", H1))
    S.append(BULLETS([
        "We generate a stream of &ldquo;pretend&rdquo; chat requests arriving over time (sometimes steady, sometimes bursty) &mdash; a realistic simulation of real traffic.",
        "We send each request at exactly its scheduled time, whether or not the system has finished earlier requests &mdash; so waiting-in-line delays are honestly captured, not hidden.",
        "A router (fixed, adaptive, or our MPI version) decides which copy of the model handles each request.",
        "That copy (running inside vLLM, using FP16, INT8, or our special INT4 kernel) actually generates the reply.",
        "We record everything: time to the first word, time for the whole reply, and how much electricity the GPU used.",
        "We repeat every experiment 8 times with identical simulated traffic each time, then use statistics to check whether any difference we see is real.",
        "Every chart and number in our slides, report, and dashboard is generated automatically by a program reading these raw recorded files &mdash; nobody types a result in by hand.",
    ]))

    # ---------------------------------------------------------------- 7 quick QA
    S.append(P("7. Rapid-fire viva answers", H1))
    pairs = [
        ("What are you trying to do, in one line?",
         "Make an AI model reply faster, by hand-writing an optimized GPU program and testing smart ways to share it among multiple requests."),
        ("Why is this a High-Performance-Computing project?",
         "It needs thousands of simultaneous calculations (GPU/CUDA), multiple machines coordinating (MPI), and careful, statistically sound measurement of all of it."),
        ("What exactly have you achieved?",
         "A verified, near-hardware-limit CUDA kernel; it is integrated into a real system (vLLM) and ties a professional kernel at one user; and a full, honest, statistically tested answer on whether smart routing helps (it did not, on one GPU, and we know why)."),
        ("What is left before the final evaluation?",
         "Testing on multiple separate GPUs (the real test of the routing idea), profiling with Nsight to directly confirm why the kernel is fast, and extending the kernel to stay fast with many simultaneous users."),
        ("Why did you pick such a small model (1.5B parameters)?",
         "Our GPU has only 8 GB of memory. A bigger, 8-billion-parameter model would not fit at all; Qwen2.5-1.5B has official versions in every precision we needed to compare."),
        ("What is the single biggest lesson from this project?",
         "Reading memory, not doing arithmetic, is what limits AI inference speed at small scale &mdash; and sharing one GPU between two things you want to compare fairly is much harder than it looks."),
        ("Is your kernel better than everything else?",
         "No, and we do not claim that. It is a tie with Marlin at one user, and it loses with many simultaneous users. We say this plainly because it is true and it is still a meaningful result."),
        ("Did anything go wrong during the project?",
         "Yes, and we kept it: an early measurement let a GPU cache inflate our bandwidth numbers by about 6%, and one library comparison was mismeasured. Both were caught by our own checks, fixed, and every affected result was re-run."),
    ]
    for q, a in pairs:
        S.append(P(q, QA_Q))
        S.append(P(a, QA_A))

    S.append(Spacer(1, 10))
    S.append(P(
        "This guide is for your own preparation and is not part of the submitted report or slides. If a question "
        "comes up that is not covered here, it is safer to say &ldquo;we have not measured that yet, and it is "
        "planned for November&rdquo; than to guess.", NOTE))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm,
                            bottomMargin=18 * mm, title="Inferno - Plain-English Guide")
    doc.build(S)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
