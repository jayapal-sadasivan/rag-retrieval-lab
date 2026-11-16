import time

import pandas as pd
import streamlit as st

import rag_core as rc
import ui

PRESETS = [
    "What is the price and warranty of the Sony WH-1000XM5?",
    "Can I return earbuds that I have already opened?",
    "How many months of warranty do chargers carry?",
    "What is the out-of-warranty battery replacement cost for the iPhone 15?",
    "Which accessories are compatible with the Dell XPS 15 9530?",
]


def _answer_card(result: dict) -> None:
    if result["abstained"]:
        st.markdown(
            f"""
            <div class="lab-callout lab-callout-warn">
              <div class="lab-callout-title">Abstained — no grounded answer</div>
              <div class="lab-callout-body">{result.get('reason', 'evidence too weak')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    citations = " ".join(f'<span class="lab-chip">{cite}</span>' for cite in result["citations"])
    scope = (f'<div style="font-size:.72rem;color:#64748b;margin-top:.5rem">'
             f'entity-scoped to <code>{result["scoped_to"]}</code></div>'
             if result.get("scoped_to") else "")
    st.markdown(
        f"""
        <div class="lab-card" style="border:1px solid var(--lab-border);border-left:4px solid #059669;
             border-radius:12px;padding:.9rem 1rem;background:var(--lab-panel)">
          <div style="font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:#64748b">
            grounded answer · confidence {result['confidence']:.2f}
          </div>
          <div style="font-size:1rem;line-height:1.55;margin:.4rem 0 .6rem 0;color:var(--lab-text)">
            {result['answer']}
          </div>
          <div class="lab-chip-row" style="margin:0">{citations}</div>
          {scope}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    engine = ui.get_engine()

    ui.page_header(
        "Grounded answers: citations, verification, abstention",
        "An answer is only trustworthy if every claim can be resolved to a retrieved span. "
        "When the evidence is missing, the system refuses instead of inventing.",
        tag="Grounding",
    )

    left, right = ui.demo_panel()
    with left:
        query = ui.query_input("ground", PRESETS, PRESETS[0])
        options = st.columns([1, 1])
        threshold = options[0].slider("Grounding threshold", 0.20, 0.60, 0.40, 0.01, key="gr_threshold")
        show_naive = options[1].toggle("Compare with an ungrounded LLM", value=True, key="gr_naive")

        started = time.perf_counter()
        result = rc.extractive_answer(engine, query, k=6, threshold=threshold)
        elapsed = (time.perf_counter() - started) * 1000
        ui.latency_chips({"hybrid": elapsed})

        _answer_card(result)

        if not result["abstained"]:
            audit = rc.verify_answer(engine, result["answer"], result["citations"])
            ok = int(audit["supported"].sum()) if len(audit) else 0
            total = len(audit)
            st.markdown(
                f"**Claim verification** — {ok}/{total} numeric and identifier claims resolved "
                f"verbatim in the cited sources."
            )
            if total:
                audit_display = audit.copy()
                audit_display["verdict"] = audit_display["supported"].map(
                    lambda value: "PASS" if value else "FAIL")
                st.dataframe(
                    audit_display[["claim", "verdict"]], hide_index=True, use_container_width=True,
                    column_config={"verdict": st.column_config.TextColumn("verified")},
                )
            else:
                st.caption("This answer contains no numeric claims to verify.")

            for citation in result["citations"]:
                text = engine.chunks.set_index("doc_id").loc[citation, "text"]
                with st.expander(f"Source {citation}", expanded=False):
                    st.code(text, language="text")

        if result["results"] is not None and len(result["results"]):
            with st.expander("Retrieved context (before generation)", expanded=False):
                ui.results_table(result["results"], score_column="rrf_score", max_rows=6)
    with right:
        ui.note_card("grounded" if not result["abstained"] else "abstain")

    if show_naive:
        ui.divider("The same question, answered from model memory")
        naive_left, naive_right = ui.demo_panel()
        with naive_left:
            naive_answer = rc.simulated_naive_generator(query)
            st.markdown(
                f"""
                <div class="lab-card" style="border:1px solid #fecaca;border-left:4px solid #dc2626;
                     border-radius:12px;padding:.9rem 1rem;background:#fff5f5">
                  <div style="font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:#991b1b">
                    simulated ungrounded answer
                  </div>
                  <div style="font-size:1rem;line-height:1.55;margin-top:.4rem;color:#111827">
                    {naive_answer}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            citation_pool = result["citations"] or [item["doc_id"] for item in result.get("context", [])]
            naive_audit = rc.verify_answer(engine, naive_answer, citation_pool)
            if len(naive_audit):
                naive_audit["verdict"] = naive_audit["supported"].map(
                    lambda value: "PASS" if value else "FAIL")
                st.dataframe(naive_audit[["claim", "verdict"]], hide_index=True,
                             use_container_width=True)
                unsupported = int((~naive_audit["supported"]).sum())
                st.error(f"{unsupported} of {len(naive_audit)} claims cannot be found in any source.")
            else:
                st.info("No numeric claims to check.")

            if not result["abstained"]:
                grounded_audit = rc.verify_answer(engine, result["answer"], result["citations"])
                comparison = pd.DataFrame([
                    {"answer": "ungrounded (simulated LLM)",
                     "claims": len(naive_audit),
                     "unsupported": int((~naive_audit["supported"]).sum()) if len(naive_audit) else 0},
                    {"answer": "grounded (extractive)",
                     "claims": len(grounded_audit),
                     "unsupported": int((~grounded_audit["supported"]).sum()) if len(grounded_audit) else 0},
                ])
                comparison["hallucination rate"] = (
                    comparison["unsupported"] / comparison["claims"].clip(lower=1)).round(2)
                st.markdown("**Side-by-side audit**")
                st.dataframe(comparison, hide_index=True, use_container_width=True)
        with naive_right:
            ui.note_card("naive")

    ui.divider("The grounding contract")
    contract_left, contract_right = st.columns([1.65, 1], gap="large")
    with contract_left:
        st.markdown(
            """
            1. **Every claim maps to a retrieved span.** No span, no claim.
            2. **Every answer carries citations** that a script can resolve.
            3. **Abstain when evidence is weak.** A confident "I don't know" beats a
               plausible invention.
            4. **Verify before returning.** Extractive answers are copied verbatim, so
               verification is a string check, not a judgement call.

            In production you replace the extractive step with an LLM and keep the same
            contract: parse JSON, verify every claim, regenerate once, then abstain.
            """
        )
        ui.callout(
            "Why abstention is a feature",
            "In support and commerce, a wrong specific number is worse than 'I don't know'. "
            "It is also a compliance liability once the answer is auditable.",
            tone="good",
        )
    with contract_right:
        ui.note_card("verify")
