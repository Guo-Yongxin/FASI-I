"use strict";

const { modelData, predictFastiOsFromRaw: predictRiskFromInput } = globalThis.FASTI ?? {};

const form = document.getElementById("risk-form");
const intervalSummary = document.getElementById("interval-summary");
const statusMessage = document.getElementById("status-message");
const resultsSection = document.getElementById("results-section");
const resultCards = document.getElementById("result-cards");
const chartContainer = document.getElementById("chart-container");
const calculationDetails = document.getElementById("calculation-details");
const fields = {
  age: form.elements.namedItem("age"),
  smoke: form.elements.namedItem("smoke"),
  tStage: form.elements.namedItem("tStage"),
  nStage: form.elements.namedItem("nStage"),
  primary: form.elements.namedItem("primary"),
  p16: form.elements.namedItem("p16"),
  rtStartDate: form.elements.namedItem("rtStartDate"),
  rtEndDate: form.elements.namedItem("rtEndDate"),
  preBloodDate: form.elements.namedItem("preBloodDate"),
  postBloodDate: form.elements.namedItem("postBloodDate"),
  hbPre: form.elements.namedItem("hbPre"),
  albPre: form.elements.namedItem("albPre"),
  lyPre: form.elements.namedItem("lyPre"),
  moPre: form.elements.namedItem("moPre"),
  hbPost: form.elements.namedItem("hbPost"),
  albPost: form.elements.namedItem("albPost"),
  lyPost: form.elements.namedItem("lyPost"),
};

const smokeLabels = {
  "0": "Never",
  "1": "Current",
  "2": "Former",
};

const primaryLabels = {
  "1": "Nasopharynx",
  "2": "Oropharynx",
  "3": "Larynx",
  "4": "Hypopharynx",
  "5": "Oral cavity",
  "6": "Other",
};

const p16Labels = {
  "0": "Negative",
  "1": "Positive",
  "2": "Not tested / unknown",
};

function parseDateInput(value) {
  const date = new Date(`${value}T00:00:00Z`);
  if (!Number.isFinite(date.getTime()) || date.toISOString().slice(0, 10) !== value) {
    throw new Error("Enter all four valid radiotherapy and blood sampling dates.");
  }
  return date;
}

function diffDays(startValue, endValue) {
  const start = parseDateInput(startValue);
  const end = parseDateInput(endValue);
  const diff = end.getTime() - start.getTime();
  return Math.round(diff / 86400000);
}

function formatFixed(value, digits = 4) {
  return Number(value).toFixed(digits);
}

function formatPercent(value, digits = 2) {
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function setStatus(message, tone = "info") {
  statusMessage.className = `status-box ${tone === "error" ? "status-error" : "status-info"}`;
  statusMessage.textContent = message;
  statusMessage.classList.remove("is-hidden");
}

function getIntervals() {
  const start = fields.rtStartDate.value;
  const end = fields.rtEndDate.value;
  const pre = fields.preBloodDate.value;
  const post = fields.postBloodDate.value;
  const interval_pre = diffDays(start, pre);
  const interval_post = diffDays(end, post);
  const dt = diffDays(pre, post);
  if (diffDays(start, end) < 0) throw new Error("RT end date cannot precede RT start date.");
  if (dt <= 0) throw new Error("The post-RT blood sample must be later than the baseline sample.");
  return { interval_pre, interval_post, dt };
}

function updateIntervalSummary() {
  try {
    const { interval_pre, interval_post, dt } = getIntervals();
    intervalSummary.innerHTML = `interval_pre: <strong>${interval_pre}</strong> days<br />interval_post*: <strong>${interval_post}</strong> days<br />Blood sampling interval: <strong>${dt}</strong> days`;
  } catch (error) {
    intervalSummary.textContent = error.message;
  }
}

function getRawInputValues() {
  return {
    ...getIntervals(),
    Primary: fields.primary.value,
    p16: fields.p16.value,
    T: fields.tStage.value,
    N: fields.nStage.value,
    Age: fields.age.value,
    Smoke: fields.smoke.value,
    HB_pre: fields.hbPre.value,
    HB_post: fields.hbPost.value,
    ALB_pre: fields.albPre.value,
    ALB_post: fields.albPost.value,
    LY_pre: fields.lyPre.value,
    LY_post: fields.lyPost.value,
    MO_pre: fields.moPre.value,
  };
}

function buildResultCard(label, value) {
  return `
    <article class="result-card">
      <div class="result-card-label">${label}</div>
      <div class="result-card-value">${value}</div>
    </article>
  `;
}

function linePath(points) {
  return points
    .map((point, index) => index === 0
      ? `M ${point.x.toFixed(2)} ${point.y.toFixed(2)}`
      : `H ${point.x.toFixed(2)} V ${point.y.toFixed(2)}`)
    .join(" ");
}

function renderChart(curve, prediction) {
  const width = 980;
  const height = 560;
  const margin = { top: 30, right: 28, bottom: 72, left: 86 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const xMax = 120;

  const xScale = (value) => margin.left + (value / xMax) * plotWidth;
  const yScale = (value) => margin.top + (1 - value) * plotHeight;

  const path = linePath(curve.map((point) => ({ x: xScale(point.time), y: yScale(point.survival) })));
  const xTicks = Array.from({ length: 11 }, (_, index) => index * 12);
  const yTicks = Array.from({ length: 6 }, (_, index) => index * 0.2);
  const markers = [
    { label: "3y / 5y", x: 36, y: 1 - prediction.OSrisk36 },
    { label: "3y / 5y", x: 60, y: 1 - prediction.OSrisk60 },
  ];

  chartContainer.innerHTML = `
    <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Individual predicted overall survival">
      ${xTicks
        .map(
          (tick) => `
            <line class="grid-line" x1="${xScale(tick)}" y1="${margin.top}" x2="${xScale(tick)}" y2="${margin.top + plotHeight}" />
            <text class="axis-tick" x="${xScale(tick)}" y="${height - margin.bottom + 28}" text-anchor="middle">${tick}</text>
          `
        )
        .join("")}
      ${yTicks
        .map(
          (tick) => `
            <line class="grid-line" x1="${margin.left}" y1="${yScale(tick)}" x2="${margin.left + plotWidth}" y2="${yScale(tick)}" />
            <text class="axis-tick" x="${margin.left - 16}" y="${yScale(tick) + 5}" text-anchor="end">${tick.toFixed(1)}</text>
          `
        )
        .join("")}
      <line class="axis-line" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}" />
      <line class="axis-line" x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${margin.left + plotWidth}" y2="${margin.top + plotHeight}" />
      <path class="curve-line" d="${path}" stroke="#1f77b4" />
      ${markers
        .map(
          (marker) => `
            <circle class="curve-marker" cx="${xScale(marker.x)}" cy="${yScale(marker.y)}" r="8.5" />
          `
        )
        .join("")}
      <g transform="translate(${margin.left}, 14)">
        <line class="legend-line" x1="0" y1="0" x2="32" y2="0" stroke="#1f77b4" />
        <text class="legend-text" x="42" y="5">Individual predicted OS</text>
        <circle class="curve-marker" cx="220" cy="0" r="7.5" />
        <text class="legend-text" x="238" y="5">3y / 5y</text>
      </g>
      <text class="axis-label" x="${margin.left + plotWidth / 2}" y="${height - 18}" text-anchor="middle">Months</text>
      <text class="axis-label" transform="translate(24 ${margin.top + plotHeight / 2}) rotate(-90)" text-anchor="middle">
        Overall survival probability
      </text>
    </svg>
  `;
}

function renderTableBlock(title, columns, row, formatters = {}, caption = "") {
  const headerHtml = columns.map((column) => `<th>${column}</th>`).join("");
  const bodyHtml = columns
    .map((column) => {
      const formatter = formatters[column];
      const value = formatter ? formatter(row[column]) : row[column];
      return `<td>${value}</td>`;
    })
    .join("");

  return `
    <section class="table-block">
      <h3 class="table-title">${title}</h3>
      <div class="table-wrap">
        <table class="centered-table">
          <thead>
            <tr>${headerHtml}</tr>
          </thead>
          <tbody>
            <tr>${bodyHtml}</tr>
          </tbody>
        </table>
      </div>
      ${caption ? `<p class="table-caption">${caption}</p>` : ""}
    </section>
  `;
}

function renderCalculationDetails(context) {
  const { raw, cleaned, predictors, primaryP16, prediction } = context;
  const rawInputs = {
    Age: Number(raw.Age),
    "Smoking history": smokeLabels[raw.Smoke],
    "Primary tumour site": primaryLabels[raw.Primary],
    p16: raw.Primary === "2" ? p16Labels[raw.p16] : "Not used",
    "T stage": `T${raw.T}`,
    "N stage": `N${raw.N}`,
    "interval_pre*": raw.interval_pre,
    "interval_post*": raw.interval_post,
    "dt*": raw.dt,
    ...Object.fromEntries(Object.keys(modelData.winsor).map((key) => [key, Number(raw[key])])),
  };

  const predictionInputs = { Primary_p16: primaryP16, ...predictors };
  const predictionOutputs = {
    Model: modelData.version,
    "RSF mortality score": prediction.mortality,
    "OS risk at 3 years": prediction.OSrisk36,
    "OS risk at 5 years": prediction.OSrisk60,
  };

  const floatColumns = (columns) =>
    Object.fromEntries(columns.map((column) => [column, (value) => formatFixed(value, 4)]));

  calculationDetails.innerHTML =
    renderTableBlock("Raw inputs", Object.keys(rawInputs), rawInputs, {}, "*, See the 'How dates and sampling intervals are defined' section for details.") +
    renderTableBlock(
      "Derived processed predictors",
      Object.keys(cleaned),
      cleaned,
      floatColumns(Object.keys(cleaned)),
      "Raw blood values are winsorised using the frozen UK 1st and 99th percentiles before computing LMR, PNI and the log post/pre changes."
    ) +
    renderTableBlock(
      "Prediction inputs",
      Object.keys(predictionInputs),
      predictionInputs,
      floatColumns(modelData.features),
      "Clinical_PI is the frozen clinical Cox linear predictor; Primary_p16 is used to compute it, not as a separate RSF input."
    ) +
    renderTableBlock(
      "Prediction outputs",
      Object.keys(predictionOutputs),
      predictionOutputs,
      {
        "RSF mortality score": (value) => formatFixed(value, 4),
        "OS risk at 3 years": (value) => formatFixed(value, 4),
        "OS risk at 5 years": (value) => formatFixed(value, 4),
      },
      "The RSF mortality score is not a probability. The locked model does not specify a clinical high/low risk cut-off."
    );
}

function renderPrediction(context) {
  resultCards.innerHTML = [
    buildResultCard("Prediction model", "M2 RSF"),
    buildResultCard("RSF mortality score", formatFixed(context.prediction.mortality, 4)),
    buildResultCard("OS risk at 3 years", formatPercent(context.prediction.OSrisk36)),
    buildResultCard("OS risk at 5 years", formatPercent(context.prediction.OSrisk60)),
  ].join("");

  renderChart(context.curve, context.prediction);
  renderCalculationDetails(context);

  resultsSection.classList.remove("is-hidden");
  statusMessage.classList.add("is-hidden");
}

function calculateRisk() {
  try {
    if (!modelData || typeof predictRiskFromInput !== "function") {
      throw new Error("The frozen RSF model could not load. Keep model.js and data/model-data.js with the website files and reload.");
    }
    const raw = getRawInputValues();
    renderPrediction({ raw, ...predictRiskFromInput(raw, modelData) });
    resultsSection.scrollIntoView({ block: "nearest" });
  } catch (error) {
    resultsSection.classList.add("is-hidden");
    setStatus(error.message, "error");
    statusMessage.scrollIntoView({ block: "nearest" });
  }
}

["rtStartDate", "rtEndDate", "preBloodDate", "postBloodDate"].forEach((fieldName) => {
  fields[fieldName].addEventListener("change", updateIntervalSummary);
});

(globalThis.FASTI ??= {}).calculateRisk = calculateRisk;
updateIntervalSummary();
