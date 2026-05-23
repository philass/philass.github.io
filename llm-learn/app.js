(function () {
  "use strict";

  const symbols = [
    ["B", "batch"],
    ["L", "number of layers"],
    ["T", "sequence length (query)"],
    ["S", "sequence length (key value)"],
    ["V", "vocab"],
    ["D", "d_model, embedding dimension"],
    ["F", "MLP hidden dimension"],
    ["H", "attention head dimension"],
    ["N", "number of query heads"],
    ["K", "number of key/value heads"],
    ["G", "q heads per kv head"],
  ].map(([symbol, answer]) => ({ symbol, answer }));

  const symbolRelationships = [
    "T often equals S during training/prefill; during cached decode T is often 1 while S grows.",
    "Standard MHA usually has N = K and G = 1; GQA/MQA usually has N > K.",
    "G = N // K, so N = K * G.",
    "D is commonly split across heads as D = N * H, so H = D // N.",
    "F is usually a multiple of D, often roughly 3D to 4D.",
    "B, L, and V are usually independent of the attention-head relationships.",
  ];

  const operations = [
    {
      name: "Attention input norm",
      prompt: "Fill in the input/output shapes and forward FLOPs for the attention norm.",
      token: "Attention input norm",
      fields: [
        ["input", "BTD", "shape"],
        ["result", "BTD", "shape"],
        ["FLOPs", "5BTD", "flops"],
      ],
      detail: "The attention norm preserves the residual-stream shape. The FLOP count is a compact LayerNorm-style estimate.",
      formula: ["input: B T D", "normalized: B T D", "FLOPs: 5 B T D"],
    },
    {
      name: "Query projection",
      prompt: "Fill in each shape and forward FLOPs for XW_Q.",
      token: "Query projection",
      fields: [
        ["X", "BTD", "shape"],
        ["W_Q", "DNH", "shape"],
        ["result", "BTNH", "shape"],
        ["FLOPs", "2BTDNH", "flops"],
      ],
      answer: "BTNH",
      detail: "The D axis contracts. B and T are batched axes; N and H are introduced by the weight.",
      formula: ["X: B T D", "W_Q: D N H", "Q: B T N H"],
    },
    {
      name: "Key projection",
      prompt: "Fill in each shape and forward FLOPs for XW_K.",
      token: "Key projection",
      fields: [
        ["X", "BSD", "shape"],
        ["W_K", "DKH", "shape"],
        ["result", "BSKH", "shape"],
        ["FLOPs", "2BSDKH", "flops"],
      ],
      answer: "BSKH",
      detail: "Keys use the key/value sequence axis S, not the query axis T.",
      formula: ["X: B S D", "W_K: D K H", "K: B S K H"],
    },
    {
      name: "Value projection",
      prompt: "Fill in each shape and forward FLOPs for XW_V.",
      token: "Value projection",
      fields: [
        ["X", "BSD", "shape"],
        ["W_V", "DKH", "shape"],
        ["result", "BSKH", "shape"],
        ["FLOPs", "2BSDKH", "flops"],
      ],
      answer: "BSKH",
      detail: "Values have the same key/value-head layout as keys.",
      formula: ["X: B S D", "W_V: D K H", "V: B S K H"],
    },
    {
      name: "Group query heads",
      prompt: "Fill in the input/output shapes and forward FLOPs for the query-head reshape.",
      token: "Group query heads",
      fields: [
        ["Q before", "BTNH", "shape"],
        ["Q after", "BTKGH", "shape"],
        ["FLOPs", "0", "flops"],
      ],
      answer: "BTKGH",
      detail: "Since G = N // K, the query-head axis N becomes K and G.",
      formula: ["Q: B T N H", "N = K G", "reshaped Q: B T K G H"],
    },
    {
      name: "Attention scores",
      prompt: "Fill in each shape and forward FLOPs for QK^T.",
      token: "Attention scores",
      fields: [
        ["Q", "BTKGH", "shape"],
        ["K", "BSKH", "shape"],
        ["result", "BTSKG", "shape"],
        ["FLOPs", "2BTSKGH", "flops"],
      ],
      answer: "BTSKG",
      detail: "The H axis contracts. The result keeps query position T, key position S, KV head K, and query group G.",
      formula: ["Q: B T K G H", "K: B S K H", "scores: B T S K G"],
    },
    {
      name: "Apply attention mask",
      prompt: "Fill in the score, mask, output shape, and forward FLOPs for applying the attention mask.",
      token: "Apply attention mask",
      fields: [
        ["scores", "BTSKG", "shape"],
        ["mask", "TS", "shape"],
        ["result", "BTSKG", "shape"],
        ["FLOPs", "BTSKG", "flops"],
      ],
      detail: "The mask is broadcast over batch and head axes, while the score tensor keeps the same shape.",
      formula: ["scores: B T S K G", "mask: T S", "masked scores: B T S K G"],
    },
    {
      name: "Attention softmax",
      prompt: "Fill in the input/output shapes and forward FLOPs for the attention softmax.",
      token: "Attention softmax",
      fields: [
        ["masked scores", "BTSKG", "shape"],
        ["result", "BTSKG", "shape"],
        ["FLOPs", "3BTSKG", "flops"],
      ],
      detail: "Softmax runs along the S axis and preserves the full attention-score shape.",
      formula: ["masked scores: B T S K G", "weights: B T S K G", "softmax axis: S"],
    },
    {
      name: "Attention weighted sum",
      prompt: "Fill in each shape and forward FLOPs for attention-weighted value mixing.",
      token: "Attention weighted sum",
      fields: [
        ["weights", "BTSKG", "shape"],
        ["V", "BSKH", "shape"],
        ["result", "BTKGH", "shape"],
        ["FLOPs", "2BTSKGH", "flops"],
      ],
      answer: "BTKGH",
      detail: "The S axis contracts when attention weights mix values over key/value positions.",
      formula: ["scores: B T S K G", "V: B S K H", "context: B T K G H"],
    },
    {
      name: "Merge query heads",
      prompt: "Fill in the input/output shapes and forward FLOPs for merging query heads.",
      token: "Merge query heads",
      fields: [
        ["context before", "BTKGH", "shape"],
        ["context after", "BTNH", "shape"],
        ["FLOPs", "0", "flops"],
      ],
      answer: "BTNH",
      detail: "K times G recovers the total number of query heads N.",
      formula: ["context: B T K G H", "K G = N", "merged: B T N H"],
    },
    {
      name: "Attention output projection",
      prompt: "Fill in each shape and forward FLOPs for the output projection.",
      token: "Attention output projection",
      fields: [
        ["attention", "BTNH", "shape"],
        ["W_O", "NHD", "shape"],
        ["result", "BTD", "shape"],
        ["FLOPs", "2BTDNH", "flops"],
      ],
      answer: "BTD",
      detail: "N and H contract, leaving the residual-stream axis D.",
      formula: ["attention: B T N H", "W_O: N H D", "output: B T D"],
    },
    {
      name: "Attention residual connection",
      prompt: "Fill in the input/output shapes and forward FLOPs for the attention residual connection.",
      token: "Attention residual connection",
      fields: [
        ["residual input", "BTD", "shape"],
        ["attention output", "BTD", "shape"],
        ["result", "BTD", "shape"],
        ["FLOPs", "BTD", "flops"],
      ],
      detail: "The attention output is added back to the residual stream elementwise.",
      formula: ["residual: B T D", "attention output: B T D", "result: B T D"],
    },
    {
      name: "MLP input norm",
      prompt: "Fill in the input/output shapes and forward FLOPs for the MLP norm.",
      token: "MLP input norm",
      fields: [
        ["input", "BTD", "shape"],
        ["result", "BTD", "shape"],
        ["FLOPs", "5BTD", "flops"],
      ],
      detail: "The MLP norm preserves the residual-stream shape. The FLOP count is a compact LayerNorm-style estimate.",
      formula: ["input: B T D", "normalized: B T D", "FLOPs: 5 B T D"],
    },
    {
      name: "MLP gate projection",
      prompt: "Fill in each shape and forward FLOPs for the MLP gate projection.",
      token: "MLP gate projection",
      fields: [
        ["X", "BTD", "shape"],
        ["W_In1", "DF", "shape"],
        ["result", "BTF", "shape"],
        ["FLOPs", "2BTDF", "flops"],
      ],
      answer: "BTF",
      detail: "The MLP expands D into the hidden dimension F.",
      formula: ["X: B T D", "W_In1: D F", "gate: B T F"],
    },
    {
      name: "MLP up projection",
      prompt: "Fill in each shape and forward FLOPs for the MLP up projection.",
      token: "MLP up projection",
      fields: [
        ["X", "BTD", "shape"],
        ["W_In2", "DF", "shape"],
        ["result", "BTF", "shape"],
        ["FLOPs", "2BTDF", "flops"],
      ],
      answer: "BTF",
      detail: "Both MLP input projections produce BTF before the elementwise product.",
      formula: ["X: B T D", "W_In2: D F", "up: B T F"],
    },
    {
      name: "MLP activation",
      prompt: "Fill in the input/output shapes and forward FLOPs for the MLP activation.",
      token: "MLP activation",
      fields: [
        ["gate", "BTF", "shape"],
        ["result", "BTF", "shape"],
        ["FLOPs", "BTF", "flops"],
      ],
      detail: "The activation is elementwise over the gate branch and keeps the MLP hidden shape.",
      formula: ["gate: B T F", "activated gate: B T F"],
    },
    {
      name: "MLP gated product",
      prompt: "Fill in the input/output shapes and forward FLOPs for the gated elementwise product.",
      token: "MLP gated product",
      fields: [
        ["activated gate", "BTF", "shape"],
        ["up", "BTF", "shape"],
        ["result", "BTF", "shape"],
        ["FLOPs", "BTF", "flops"],
      ],
      detail: "The activated gate and up branch are multiplied elementwise.",
      formula: ["activated gate: B T F", "up: B T F", "product: B T F"],
    },
    {
      name: "MLP output projection",
      prompt: "Fill in each shape and forward FLOPs for the MLP output projection.",
      token: "MLP output projection",
      fields: [
        ["activation", "BTF", "shape"],
        ["W_Out", "FD", "shape"],
        ["result", "BTD", "shape"],
        ["FLOPs", "2BTDF", "flops"],
      ],
      answer: "BTD",
      detail: "The hidden axis F contracts back down to D.",
      formula: ["activation: B T F", "W_Out: F D", "output: B T D"],
    },
    {
      name: "MLP residual connection",
      prompt: "Fill in the input/output shapes and forward FLOPs for the MLP residual connection.",
      token: "MLP residual connection",
      fields: [
        ["residual input", "BTD", "shape"],
        ["MLP output", "BTD", "shape"],
        ["result", "BTD", "shape"],
        ["FLOPs", "BTD", "flops"],
      ],
      detail: "The MLP output is added back to the residual stream elementwise.",
      formula: ["residual: B T D", "MLP output: B T D", "result: B T D"],
    },
  ];

  const state = {
    index: 0,
    stats: loadStats(),
  };

  const $ = (selector) => document.querySelector(selector);
  const els = {
    dateLabel: $("#dateLabel"),
    accuracy: $("#accuracy"),
    answered: $("#answered"),
    streak: $("#streak"),
    modeLabel: $("#modeLabel"),
    promptTitle: $("#promptTitle"),
    questionKicker: $("#questionKicker"),
    bigToken: $("#bigToken"),
    questionText: $("#questionText"),
    symbolReference: $("#symbolReference"),
    answerToolbar: $("#answerToolbar"),
    symbolToggle: $("#symbolToggle"),
    answerFields: $("#answerFields"),
    answerForm: $("#answerForm"),
    feedback: $("#feedback"),
    scoreBadge: $("#scoreBadge"),
    feedbackTitle: $("#feedbackTitle"),
    feedbackText: $("#feedbackText"),
    shapeBox: $("#shapeBox"),
    nextButton: $("#nextButton"),
  };

  function makeOperationCards() {
    return operations.map((item) => ({
      kicker: item.name,
      token: item.token,
      prompt: item.prompt,
      fields: item.fields,
      detail: item.detail,
      formula: item.formula,
    }));
  }

  const cards = makeOperationCards();

  function loadStats() {
    try {
      return JSON.parse(localStorage.getItem("transformerAccountingStats")) || defaultStats();
    } catch (error) {
      return defaultStats();
    }
  }

  function defaultStats() {
    return { correct: 0, answered: 0, streak: 0 };
  }

  function saveStats() {
    localStorage.setItem("transformerAccountingStats", JSON.stringify(state.stats));
  }

  function currentCard() {
    return cards[state.index];
  }

  function render() {
    els.dateLabel.textContent = new Date().toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
    els.modeLabel.textContent = "Operations";
    els.promptTitle.textContent = "What shape comes out?";
    els.nextButton.hidden = false;

    const card = currentCard();
    els.questionKicker.textContent = card.kicker;
    els.bigToken.textContent = card.token;
    els.bigToken.hidden = false;
    els.bigToken.classList.add("compact");
    els.questionText.innerHTML = renderMathText(card.prompt);
    els.feedback.hidden = true;
    els.answerToolbar.hidden = false;
    els.symbolReference.hidden = !els.symbolToggle.checked;
    els.symbolReference.innerHTML = renderSymbolTable();
    els.answerFields.innerHTML = card.fields
      .map(
        ([label, , type], index) => `
          <label class="answer-field">
            <span>${renderMathText(label)}</span>
            <input
              type="text"
              name="field-${index}"
              autocomplete="off"
              autocapitalize="characters"
              spellcheck="false"
              placeholder="${placeholderFor(type)}"
            />
          </label>
        `,
      )
      .join("");

    renderStats();
  }

  function renderStats() {
    const pct = state.stats.answered ? Math.round((state.stats.correct / state.stats.answered) * 100) + "%" : "New";
    els.accuracy.textContent = pct;
    els.answered.textContent = state.stats.answered;
    els.streak.textContent = state.stats.streak;
  }

  function submit(event) {
    event.preventDefault();
    const card = currentCard();
    const form = new FormData(els.answerForm);
    const results = card.fields.map(([label, expected, type], index) => {
      const rawGuess = form.get(`field-${index}`) || "";
      const guess = normalizeAnswer(rawGuess, type);
      const target = normalizeAnswer(expected, type);
      return { label, expected, guess: String(rawGuess), correct: guess === target };
    });
    const correctCount = results.filter((item) => item.correct).length;
    const correct = correctCount === results.length;
    const score = Math.round((correctCount / results.length) * 100);

    state.stats.answered += 1;
    state.stats.correct += correct ? 1 : 0;
    state.stats.streak = correct ? state.stats.streak + 1 : 0;
    saveStats();
    renderStats();

    els.feedback.hidden = false;
    els.scoreBadge.textContent = score;
    els.scoreBadge.style.borderColor = correct ? "var(--green)" : "var(--red)";
    els.feedbackTitle.textContent = correct ? "Correct" : `Correct fields: ${correctCount}/${results.length}`;
    els.feedbackText.textContent = card.detail;
    els.shapeBox.innerHTML = [
      ...results.map((item) => {
        const mark = item.correct ? "OK" : "MISS";
        return `${mark} ${escapeHtml(item.label)}: ${colorShape(item.expected)}`;
      }),
      ...card.formula.map((line) => colorShape(line)),
    ]
      .map((line) => `<div>${line}</div>`)
      .join("");
  }

  function placeholderFor(type) {
    if (type === "flops") return "e.g. 2BTDNH";
    return "shape";
  }

  function normalizeAnswer(value, type) {
    const trimmed = String(value).trim();
    return trimmed
      .toUpperCase()
      .replace(/\s+/g, "")
      .replaceAll("*", "")
      .replaceAll(".", "")
      .replaceAll("X", "")
      .replaceAll("FLOPS", "")
      .replaceAll("FLOP", "")
      .replaceAll("->", ">");
  }

  function renderSymbolTable() {
    const table = symbols
      .map((item) => `<div><strong>${escapeHtml(item.symbol)}</strong><span>${escapeHtml(item.answer)}</span></div>`)
      .join("");
    const relationships = symbolRelationships
      .map((item) => `<li>${escapeHtml(item)}</li>`)
      .join("");

    return `
      <div class="symbol-reference-grid">${table}</div>
      <section class="symbol-relationships" aria-label="Symbol relationships">
        <h4>Relationships</h4>
        <ul>${relationships}</ul>
      </section>
    `;
  }

  function renderMathText(value) {
    return escapeHtml(value)
      .replace(/W_([A-Za-z0-9]+)/g, "W<sub>$1</sub>")
      .replace(/\^T/g, "<sup>T</sup>")
      .replace(/XW<sub>([A-Za-z0-9]+)<\/sub>/g, "XW<sub>$1</sub>");
  }

  function colorShape(line) {
    return escapeHtml(line).replace(/[BTSDLVFHNKG]+/g, (match) => {
      const colored = match
        .split("")
        .map((letter) => {
          const className = ["D", "H", "F"].includes(letter) ? "red" : "blue";
          return `<span class="${className}">${letter}</span>`;
        })
        .join("");
      return colored;
    });
  }

  function next() {
    state.index = (state.index + 1) % cards.length;
    render();
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  els.answerForm.addEventListener("submit", submit);
  els.nextButton.addEventListener("click", next);
  els.symbolToggle.addEventListener("change", () => {
    els.symbolReference.hidden = !els.symbolToggle.checked;
  });
  render();
})();
