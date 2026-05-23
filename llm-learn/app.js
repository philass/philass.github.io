(function () {
  "use strict";

  const symbols = [
    ["B", "batch", "How many examples are processed together."],
    ["L", "number of layers", "How many decoder blocks are stacked."],
    ["T", "sequence length (query)", "The query-token axis in the current forward pass."],
    ["S", "sequence length (key value)", "The key/value-token axis attended over."],
    ["V", "vocab", "The vocabulary axis used by token embeddings and logits."],
    ["D", "d_model, embedding dimension", "The residual-stream width."],
    ["F", "MLP hidden dimension", "The expanded hidden axis inside the feed-forward block."],
    ["H", "attention head dimension", "The width of each attention head."],
    ["N", "number of query heads", "How many query heads the model has."],
    ["K", "number of key/value heads", "How many shared key/value heads are used."],
    ["G", "q heads per kv head = N // K", "Grouped-query attention ratio."],
  ].map(([symbol, answer, note]) => ({ symbol, answer, note }));

  const operations = [
    {
      name: "Query projection",
      prompt: "Fill in each shape and forward FLOPs for XW_Q.",
      token: "BTD x DNH",
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
      token: "BSD x DKH",
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
      token: "BSD x DKH",
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
      token: "BTNH reshape",
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
      token: "QK^T",
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
      name: "Attention weighted sum",
      prompt: "Fill in each shape and forward FLOPs for attention-weighted value mixing.",
      token: "scores x V",
      fields: [
        ["scores", "BTSKG", "shape"],
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
      token: "BTKGH reshape",
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
      name: "Output projection",
      prompt: "Fill in each shape and forward FLOPs for the output projection.",
      token: "BTNH x NHD",
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
      name: "MLP gate projection",
      prompt: "Fill in each shape and forward FLOPs for the MLP gate projection.",
      token: "BTD x DF",
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
      token: "BTD x DF",
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
      name: "MLP output projection",
      prompt: "Fill in each shape and forward FLOPs for the MLP output projection.",
      token: "BTF x FD",
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
  ];

  const paths = [
    {
      name: "Attention full path",
      token: "X -> Attention -> residual",
      prompt: "Choose the correct forward-pass shape path through attention.",
      answer: "BTD -> BTNH -> BTKGH -> BTSKG -> BTKGH -> BTNH -> BTD",
      detail: "This is the core attention route: project, group heads, score, mix values, merge heads, project out.",
      formula: [
        "BTD x DNH = BTNH",
        "BTNH -> BTKGH",
        "BTKGH x BSKH = BTSKG",
        "BTSKG x BSKH = BTKGH",
        "BTKGH -> BTNH",
        "BTNH x NHD = BTD",
      ],
    },
    {
      name: "MLP full path",
      token: "X -> MLP -> residual",
      prompt: "Choose the correct forward-pass shape path through the gated MLP.",
      answer: "BTD -> BTF and BTF -> BTF -> BTD",
      detail: "Both input projections expand to F; the gated activation stays BTF before projecting back to D.",
      formula: ["BTD x DF = BTF", "gelu(BTF) * BTF = BTF", "BTF x FD = BTD"],
    },
    {
      name: "Grouped-query attention",
      token: "N, K, G",
      prompt: "Choose the correct relationship for grouped-query attention in this notation.",
      answer: "N = K * G",
      detail: "G is query heads per key/value head, so G = N // K and N = K * G.",
      formula: ["N: query heads", "K: key/value heads", "G: query heads per KV head"],
    },
  ];

  const tidbits = [
    {
      title: "T = S",
      label: "Training and prefill",
      text:
        "In full-sequence self-attention during training or prefill, query length and key/value length are usually the same sequence length.",
      note: "During cached decoding, T is often 1 while S is the context length so far.",
    },
    {
      title: "N = K",
      label: "Standard MHA",
      text:
        "In standard multi-head attention, the number of query heads usually equals the number of key/value heads.",
      note: "Grouped-query attention uses K < N; multi-query attention is the extreme case where K = 1.",
    },
    {
      title: "G = 1",
      label: "No grouping",
      text: "When N = K, each key/value head serves one query head, so G = N // K = 1.",
      note: "When K is smaller than N, G tells you how many query heads share each key/value head.",
    },
    {
      title: "D = N * H",
      label: "Attention width",
      text:
        "The model dimension is commonly split evenly across attention heads, so total attention width equals query heads times head dimension.",
      note: "This makes reshaping between BTD and BTNH feel natural.",
    },
    {
      title: "F ~= 3D to 4D-ish",
      label: "MLP hidden size",
      text:
        "Classic Transformer MLPs often use about 4D hidden width, while modern gated MLPs often land closer to a different multiple.",
      note: "F is related to D by design choice, but it is not the same kind of axis as sequence length or batch size.",
    },
    {
      title: "T << S",
      label: "Inference decode",
      text:
        "During token-by-token cached decoding, T can be just the new token count while S grows with the accumulated context.",
      note: "This is why decode attention has a different shape feel from training attention.",
    },
  ];

  const modeConfig = {
    symbols: {
      label: "Symbols",
      title: "What does this symbol mean?",
      cards: makeSymbolCards(),
    },
    operations: {
      label: "Operations",
      title: "What shape comes out?",
      cards: makeOperationCards(),
    },
    path: {
      label: "Path Recall",
      title: "Can you reconstruct the path?",
      cards: makePathCards(),
    },
    tidbits: {
      label: "Tidbits",
      title: "Which symbols are usually equal?",
      cards: [],
    },
  };

  const state = {
    mode: "symbols",
    indexes: { symbols: 0, operations: 0, path: 0 },
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
    tabs: document.querySelectorAll(".tab"),
  };

  function makeSymbolCards() {
    return symbols.map((item) => ({
      kicker: "Symbol",
      token: item.symbol,
      prompt: "Type exactly what this symbol corresponds to.",
      fields: [[item.symbol, item.answer, "text"]],
      detail: `${item.symbol} means ${item.answer}. ${item.note}`,
      formula: [`${item.symbol} = ${item.answer}`],
    }));
  }

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

  function makePathCards() {
    return paths.map((item) => ({
      kicker: item.name,
      token: item.token,
      prompt: item.prompt.replace("Choose", "Type"),
      fields: [[item.name, item.answer, "path"]],
      detail: item.detail,
      formula: item.formula,
    }));
  }

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
    if (state.mode === "tidbits") return null;
    return modeConfig[state.mode].cards[state.indexes[state.mode]];
  }

  function render() {
    const mode = modeConfig[state.mode];

    els.dateLabel.textContent = new Date().toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
    els.modeLabel.textContent = mode.label;
    els.promptTitle.textContent = mode.title;
    els.tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.mode === state.mode));
    els.nextButton.hidden = state.mode === "tidbits";
    els.answerForm.classList.toggle("reference-mode", state.mode === "tidbits");

    if (state.mode === "tidbits") {
      renderTidbits();
      renderStats();
      return;
    }

    const card = currentCard();
    els.questionKicker.textContent = card.kicker;
    els.bigToken.textContent = card.token;
    els.bigToken.hidden = state.mode !== "symbols";
    els.bigToken.classList.toggle("compact", state.mode !== "symbols");
    els.questionText.innerHTML = renderMathText(card.prompt);
    els.feedback.hidden = true;
    els.answerToolbar.hidden = state.mode !== "operations";
    els.symbolReference.hidden = state.mode !== "operations" || !els.symbolToggle.checked;
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

  function renderTidbits() {
    els.questionKicker.textContent = "Defaults";
    els.bigToken.hidden = true;
    els.questionText.innerHTML =
      "Common equalities are useful mental shortcuts, but keep the symbols separate because training, prefill, grouped-query attention, and decode can differ.";
    els.answerToolbar.hidden = true;
    els.symbolReference.hidden = true;
    els.answerFields.innerHTML = `
      <div class="tidbit-grid">
        ${tidbits
          .map(
            (item) => `
              <section class="tidbit">
                <div>
                  <strong>${colorShape(item.title)}</strong>
                  <span>${escapeHtml(item.label)}</span>
                </div>
                <p>${escapeHtml(item.text)}</p>
                <small>${escapeHtml(item.note)}</small>
              </section>
            `,
          )
          .join("")}
      </div>
    `;
    els.answerForm.classList.add("reference-mode");
    els.feedback.hidden = false;
    els.scoreBadge.textContent = "!";
    els.scoreBadge.style.borderColor = "var(--teal)";
    els.feedbackTitle.textContent = "Mental Defaults";
    els.feedbackText.textContent =
      "Training self-attention often has T = S. Standard MHA often has N = K and G = 1. Attention width is commonly D = N * H. During inference decode, T is often much smaller than S.";
    els.shapeBox.innerHTML = ["B is independent", "L is independent", "V is usually much larger", "F is usually a multiple of D"]
      .map((line) => `<div>${colorShape(line)}</div>`)
      .join("");
  }

  function renderStats() {
    const pct = state.stats.answered ? Math.round((state.stats.correct / state.stats.answered) * 100) + "%" : "New";
    els.accuracy.textContent = pct;
    els.answered.textContent = state.stats.answered;
    els.streak.textContent = state.stats.streak;
  }

  function submit(event) {
    event.preventDefault();
    if (state.mode === "tidbits") return;
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
    if (type === "text") return "exact meaning";
    if (type === "flops") return "e.g. 2BTDNH";
    if (type === "path") return "full path";
    return "shape";
  }

  function normalizeAnswer(value, type) {
    const trimmed = String(value).trim();
    if (type === "text") {
      return trimmed.toLowerCase().replace(/\s+/g, " ");
    }
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
    return symbols
      .map((item) => `<div><strong>${escapeHtml(item.symbol)}</strong><span>${escapeHtml(item.answer)}</span></div>`)
      .join("");
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
    if (state.mode === "tidbits") return;
    const cards = modeConfig[state.mode].cards;
    state.indexes[state.mode] = (state.indexes[state.mode] + 1) % cards.length;
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

  function escapeAttr(value) {
    return escapeHtml(value);
  }

  els.tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      state.mode = tab.dataset.mode;
      render();
    });
  });

  els.answerForm.addEventListener("submit", submit);
  els.nextButton.addEventListener("click", next);
  els.symbolToggle.addEventListener("change", () => {
    els.symbolReference.hidden = state.mode !== "operations" || !els.symbolToggle.checked;
  });
  render();
})();
