"use strict";

const primarySites = Object.freeze({
  1: "NPC", 2: "OPC", 3: "Larynx", 4: "Hypopharynx", 5: "Oral_cavity", 6: "Other",
});
const opcP16 = Object.freeze({ 0: "negative", 1: "positive", 2: "unknown" });

function numericInput(value, name) {
  if (value == null || String(value).trim() === "" || !Number.isFinite(Number(value))) {
    throw new Error(`Enter a valid value for ${name}.`);
  }
  return Number(value);
}

function level(value, allowed, name) {
  const code = String(value ?? "").trim();
  if (!allowed.includes(code)) throw new Error(`Select a valid ${name}.`);
  return code;
}

function lockedPrecision(value) {
  // At nine decimals, an exact binary midpoint is an odd multiple of 1/1024.
  const midpoint = Math.abs(value) * 1024;
  if (Number.isInteger(midpoint) && midpoint % 2 === 1) {
    const lower = Math.floor(Math.abs(value) * 1e9);
    return Math.sign(value) * (lower + lower % 2) / 1e9;
  }
  return Number(value.toFixed(9));
}

function makePrimaryP16(primary, p16) {
  const site = primarySites[level(primary, Object.keys(primarySites), "primary tumour site")];
  if (site !== "OPC") return site;
  const code = String(p16 ?? "").trim() || "2";
  return `OPC_p16_${opcP16[level(code, Object.keys(opcP16), "p16 status")]}`;
}

function prepareFastiInput(raw, data) {
  const age = numericInput(raw.Age, "age");
  if (age < 18 || age > 100) throw new Error("Age must be between 18 and 100 years.");
  const clinical = { Age: age, Primary_p16: makePrimaryP16(raw.Primary, raw.p16) };
  let clinicalPi = age * Number(data.clinical.ageCoefficient);
  for (const [name, coefficients] of Object.entries(data.clinical.factors)) {
    const code = level(clinical[name] ?? raw[name], Object.keys(coefficients), name);
    clinicalPi += Number(coefficients[code]);
  }

  const cleaned = {};
  for (const [name, [lo, hi]] of Object.entries(data.winsor)) {
    const value = numericInput(raw[name], name);
    if (value < 0 || (/^(HB|ALB)_/.test(name) && value === 0)) {
      throw new Error(`Enter a valid measured value for ${name}.`);
    }
    cleaned[`${name}_w`] = Math.min(Math.max(value, Number(lo)), Number(hi));
  }
  const pniPre = cleaned.ALB_pre_w + 5 * cleaned.LY_pre_w;
  const pniPost = cleaned.ALB_post_w + 5 * cleaned.LY_post_w;
  Object.assign(cleaned, {
    LMR_pre_w: lockedPrecision(cleaned.LY_pre_w / cleaned.MO_pre_w),
    PNI_pre_w: lockedPrecision(pniPre),
    PNI_post_w: lockedPrecision(pniPost),
    PNI_L_w: lockedPrecision(Math.log(pniPost / pniPre)),
    HB_L_w: lockedPrecision(Math.log(cleaned.HB_post_w / cleaned.HB_pre_w)),
    HB_pre_w: lockedPrecision(cleaned.HB_pre_w),
  });
  const predictors = {
    Clinical_PI: clinicalPi,
    interval_pre: numericInput(raw.interval_pre, "pretreatment interval"),
    HB_pre_w: cleaned.HB_pre_w,
    LMR_pre_w: cleaned.LMR_pre_w,
    PNI_pre_w: cleaned.PNI_pre_w,
    interval_post: numericInput(raw.interval_post, "post-treatment interval"),
    PNI_L_w: cleaned.PNI_L_w,
    HB_L_w: cleaned.HB_L_w,
  };
  return { cleaned, predictors, primaryP16: clinical.Primary_p16 };
}

function predictFastiRsf(predictors, data) {
  const x = data.features.map((name) => numericInput(predictors[name], name));
  const totals = new Float64Array(data.maxMonth + 1);
  let mortality = 0;
  for (const root of data.roots) {
    let index = root;
    let node = data.nodes[index];
    // Preorder storage: left child follows the split node; right child is explicit.
    while (node[0] >= 0) {
      index = x[node[0]] <= Number(node[2]) ? index + 1 : node[1];
      node = data.nodes[index];
    }

    const leaf = data.leaves[node[1]];
    mortality += leaf[0];
    let survival = 1;
    let j = 1;
    for (let month = 0; month <= data.maxMonth; month += 1) {
      if (month === leaf[j]) {
        survival = leaf[j + 1];
        j += 2;
      }
      totals[month] += survival;
    }
  }
  const curve = Array.from(totals, (total, time) => {
    const survival = total / data.roots.length;
    return { time, survival, risk: 1 - survival };
  });
  return {
    curve,
    prediction: {
      mortality: mortality / data.roots.length,
      OSrisk36: curve[36].risk,
      OSrisk60: curve[60].risk,
    },
  };
}

function predictFastiOsFromRaw(raw, data) {
  const prepared = prepareFastiInput(raw, data);
  return { ...prepared, ...predictFastiRsf(prepared.predictors, data) };
}

Object.assign(globalThis.FASTI ??= {}, { makePrimaryP16, predictFastiOsFromRaw, predictFastiRsf });
