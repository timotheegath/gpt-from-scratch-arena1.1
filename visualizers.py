import pandas as pd
import plotly.express as px  # type: ignore
import torch as t
from jaxtyping import Int
from torch import Tensor
from transformers import PreTrainedTokenizerBase

# ruff: noqa: F722


def display_logits(
    tokenizer: PreTrainedTokenizerBase,
    tokens: list[int],
    logits: Int[Tensor, "batch position d_vocab"],
) -> None:
    # Extract top-10 for each position
    records = []
    # Only considering the first batch

    for pos in range(len(tokens)):
        if tokenizer is None:
            raise TypeError
        token_id = tokens[pos]
        token_text = tokenizer.decode(token_id)

        probs = t.softmax(logits[0, pos, :], dim=-1)  # Only considering t he first batch
        top_probs, top_ids = t.topk(probs, 10)

        for rank, (prob, tid) in enumerate(zip(top_probs, top_ids, strict=False), 1):
            records.append(
                {
                    "Position": pos,
                    "Input Token": token_text,
                    "Rank": rank,
                    "Predicted Token": tokenizer.decode(tid),
                    "Probability": prob.item(),
                }
            )

    df = pd.DataFrame(records)

    # Create grouped bar chart
    fig = px.bar(
        df,
        x="Input Token",
        y="Probability",
        color="Rank",
        barmode="group",
        custom_data=["Position", "Predicted Token", "Rank"],
        color_discrete_sequence=px.colors.sequential.Blues_r,
        title="Top-10 Next-Token Probabilities for Each Input Token",
        labels={"Probability": "Probability", "Input Token": "Input Token Position"},
    )

    fig.update_layout(
        xaxis_title="Input Token (position)",
        yaxis_title="Probability of Next Token",
        legend_title="Rank in Top-10",
        hovermode="closest",
        height=600,
    )

    fig.update_traces(
        texttemplate="%{y:.2%}",
        textposition="outside",
        hovertemplate=(
            "Input token: %{x}<br>"
            "Position: %{customdata[0]}<br>"
            "Predicted token: %{customdata[1]}<br>"
            "Rank: %{customdata[2]}<br>"
            "Probability: %{y:.4%}<extra></extra>"
        ),
    )

    fig.write_html("top10_logits_per_token.html")
