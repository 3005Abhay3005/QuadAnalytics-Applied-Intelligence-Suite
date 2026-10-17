/**
 * Quad Analytics — Core Client Application Engine
 * Pure Applied Machine Learning Platform
 * Zero User Telemetry / No Personal Data Retention / Zero Emojis
 */

const THEME_STORAGE_KEY = "quad_theme_preference";

// ─── Theme Management ────────────────────────────────────────────────────────

function initTheme() {
  const saved = localStorage.getItem(THEME_STORAGE_KEY) || "light";
  applyTheme(saved);

  const toggleBtn = document.getElementById("themeToggle");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme") || "light";
      const next = current === "dark" ? "light" : "dark";
      localStorage.setItem(THEME_STORAGE_KEY, next);
      applyTheme(next);
    });
  }
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const sun = document.getElementById("iconSun");
  const moon = document.getElementById("iconMoon");
  if (sun && moon) {
    if (theme === "dark") {
      sun.style.display = "none";
      moon.style.display = "inline";
    } else {
      sun.style.display = "inline";
      moon.style.display = "none";
    }
  }

  if (window.Chart) {
    const textColor = theme === "dark" ? "#A0A0A0" : "#5A5A5A";
    const borderColor = theme === "dark" ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.08)";
    Chart.defaults.color = textColor;
    Chart.defaults.borderColor = borderColor;
  }
}

// ─── Mobile Navigation ───────────────────────────────────────────────────────

function initMobileNav() {
  const menuBtn = document.getElementById("menuBtn");
  const mobileNav = document.getElementById("mobileNav");
  if (menuBtn && mobileNav) {
    menuBtn.addEventListener("click", () => {
      mobileNav.classList.toggle("open");
    });
  }
}

// ─── Synchronized Range Sliders (No color shifts) ────────────────────────────

function initRangeSync() {
  document.querySelectorAll(".slider-row").forEach((row) => {
    const slider = row.querySelector("input[type='range']");
    const number = row.querySelector("input[type='number']");
    if (!slider || !number) return;

    slider.addEventListener("input", () => {
      number.value = slider.value;
    });

    number.addEventListener("input", () => {
      const val = parseFloat(number.value);
      if (!isNaN(val)) {
        slider.value = val;
      }
    });
  });
}

// ─── Chart.js Helper Utilities ───────────────────────────────────────────────

let resultChartInstance = null;
let attributionChartInstance = null;

function renderProbabilityDoughnut(canvasId, labels, data, colors) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  if (resultChartInstance) resultChartInstance.destroy();

  resultChartInstance = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: colors,
        borderWidth: 0,
        hoverOffset: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "72%",
      plugins: {
        legend: {
          position: "right",
          labels: { boxWidth: 12, font: { size: 12 } }
        },
        tooltip: {
          callbacks: {
            label: (item) => ` ${item.label}: ${item.raw}%`
          }
        }
      }
    }
  });
}

function renderValueBar(canvasId, label, value, min, max, barColor) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  if (resultChartInstance) resultChartInstance.destroy();

  resultChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: [label],
      datasets: [{
        data: [value],
        backgroundColor: barColor || "rgba(15, 15, 15, 0.8)",
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { min, max, grid: { color: "rgba(150, 150, 140, 0.12)" } },
        y: { display: false }
      }
    }
  });
}

function renderAttributions(canvasId, attributions) {
  const ctx = document.getElementById(canvasId);
  if (!ctx || !attributions || !attributions.length) return;
  if (attributionChartInstance) attributionChartInstance.destroy();

  const labels = attributions.map((a) => a.feature);
  const data = attributions.map((a) => a.impact);
  const colors = attributions.map((a) =>
    a.direction === "positive" ? "rgba(220, 38, 38, 0.8)" : "rgba(22, 163, 74, 0.8)"
  );

  attributionChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: colors,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (item) => ` Contribution: ${item.raw > 0 ? "+" : ""}${item.raw}`
          }
        }
      },
      scales: {
        x: { grid: { color: "rgba(150, 150, 140, 0.12)" } },
        y: { grid: { display: false } }
      }
    }
  });
}

// ─── UI Helper Routines ──────────────────────────────────────────────────────

function setBtnLoading(btn, loading) {
  if (!btn) return;
  btn.classList.toggle("loading", loading);
  btn.disabled = loading;
}

function showApiError(msg) {
  const errBox = document.getElementById("api-error");
  if (!errBox) return;
  errBox.textContent = msg;
  errBox.classList.add("show");
}

function hideApiError() {
  const errBox = document.getElementById("api-error");
  if (errBox) errBox.classList.remove("show");
}

function revealResultPanel() {
  const empty = document.getElementById("result-empty");
  const body = document.getElementById("result-body");
  if (empty) empty.style.display = "none";
  if (body) body.classList.add("visible");
}

async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error (HTTP ${res.status})`);
  }
  return res.json();
}

// ─── Model 01: Diabetes ──────────────────────────────────────────────────────

const DIABETES_PRESETS = {
  healthy: { pregnancies: 1, glucose: 88, blood_pressure: 70, insulin: 15, bmi: 21.4, age: 24 },
  borderline: { pregnancies: 3, glucose: 118, blood_pressure: 82, insulin: 45, bmi: 27.2, age: 38 },
  high_risk: { pregnancies: 6, glucose: 168, blood_pressure: 92, insulin: 180, bmi: 35.8, age: 52 }
};

function initDiabetes() {
  const form = document.getElementById("diabetes-form");
  const submitBtn = document.getElementById("diabetes-submit");
  if (!form) return;

  function applyPreset(p) {
    Object.keys(p).forEach((k) => {
      const num = document.getElementById(k);
      const slider = document.getElementById(`slider-${k}`);
      if (num) num.value = p[k];
      if (slider) slider.value = p[k];
    });
  }

  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const p = DIABETES_PRESETS[btn.dataset.preset];
      if (p) applyPreset(p);
    });
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideApiError();
    setBtnLoading(submitBtn, true);

    try {
      const body = {
        pregnancies: parseFloat(document.getElementById("pregnancies").value),
        glucose: parseFloat(document.getElementById("glucose").value),
        blood_pressure: parseFloat(document.getElementById("blood_pressure").value),
        insulin: parseFloat(document.getElementById("insulin").value),
        bmi: parseFloat(document.getElementById("bmi").value),
        age: parseFloat(document.getElementById("age").value)
      };

      const data = await postJSON("/api/predict/diabetes", body);
      const pred = data.prediction;

      document.getElementById("result-value").textContent = pred.result;
      document.getElementById("result-risk").textContent = `${pred.risk_level} Risk Level Assessment`;
      document.getElementById("result-confidence").textContent = `Confidence: ${pred.confidence}%`;

      const isDiabetic = pred.label === 1;
      const probPos = isDiabetic ? pred.confidence : 100 - pred.confidence;
      const probNeg = 100 - probPos;

      renderProbabilityDoughnut(
        "result-chart",
        ["Diabetic Risk", "Non-Diabetic"],
        [probPos, probNeg],
        ["rgba(220, 38, 38, 0.8)", "rgba(22, 163, 74, 0.8)"]
      );

      if (pred.attributions) {
        renderAttributions("attribution-chart", pred.attributions);
      }

      revealResultPanel();
    } catch (err) {
      showApiError(err.message);
    } finally {
      setBtnLoading(submitBtn, false);
    }
  });
}

// ─── Model 02: Breast Cancer ─────────────────────────────────────────────────

const CANCER_PRESETS = {
  healthy: { Age: 26, BMI: 21.0, Smoking: 0, GeneticRisk: 0, PhysicalActivity: 6.5, AlcoholIntake: 1.0, CancerHistory: 0 },
  moderate: { Age: 45, BMI: 28.5, Smoking: 0, GeneticRisk: 1, PhysicalActivity: 2.0, AlcoholIntake: 8.0, CancerHistory: 0 },
  high_risk: { Age: 62, BMI: 34.2, Smoking: 1, GeneticRisk: 2, PhysicalActivity: 0.5, AlcoholIntake: 18.0, CancerHistory: 1 }
};

function initBreastCancer() {
  const form = document.getElementById("cancer-form");
  const submitBtn = document.getElementById("cancer-submit");
  if (!form) return;

  function applyPreset(p) {
    Object.keys(p).forEach((k) => {
      const el = document.getElementById(k);
      const slider = document.getElementById(`slider-${k}`);
      if (el) el.value = p[k];
      if (slider) slider.value = p[k];
    });
  }

  document.querySelectorAll("[data-cancer-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const p = CANCER_PRESETS[btn.dataset.cancerPreset];
      if (p) applyPreset(p);
    });
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideApiError();
    setBtnLoading(submitBtn, true);

    try {
      const body = {
        Age: parseFloat(document.getElementById("Age").value),
        BMI: parseFloat(document.getElementById("BMI").value),
        Smoking: parseInt(document.getElementById("Smoking").value, 10),
        GeneticRisk: parseInt(document.getElementById("GeneticRisk").value, 10),
        PhysicalActivity: parseFloat(document.getElementById("PhysicalActivity").value),
        AlcoholIntake: parseFloat(document.getElementById("AlcoholIntake").value),
        CancerHistory: parseInt(document.getElementById("CancerHistory").value, 10)
      };

      const data = await postJSON("/api/predict/breast-cancer", body);
      const pred = data.prediction;

      const isPos = pred.label === 1;
      document.getElementById("result-value").textContent = isPos ? "High Risk Detected" : "Low Risk Profile";
      document.getElementById("result-risk").textContent = `${pred.risk_level} Clinical Stratification`;
      document.getElementById("result-confidence").textContent = `Confidence: ${pred.confidence}%`;

      const probHigh = isPos ? pred.confidence : 100 - pred.confidence;
      const probLow = 100 - probHigh;

      renderProbabilityDoughnut(
        "result-chart",
        ["High Risk Probability", "Low Risk Baseline"],
        [probHigh, probLow],
        ["rgba(220, 38, 38, 0.8)", "rgba(22, 163, 74, 0.8)"]
      );

      if (pred.attributions) {
        renderAttributions("attribution-chart", pred.attributions);
      }

      revealResultPanel();
    } catch (err) {
      showApiError(err.message);
    } finally {
      setBtnLoading(submitBtn, false);
    }
  });
}

// ─── Model 03: Car Price ─────────────────────────────────────────────────────

const CAR_SAMPLE = {
  company: "Hyundai",
  name: "Hyundai Creta",
  year: 2018,
  kms_driven: 42000,
  fuel_type: "Diesel",
  transmission: "Manual",
  location: "Mumbai"
};

async function initCarPrice() {
  const form = document.getElementById("car-form");
  const submitBtn = document.getElementById("car-submit");
  const companySel = document.getElementById("company");
  const nameSel = document.getElementById("name");
  const fuelSel = document.getElementById("fuel_type");
  const transSel = document.getElementById("transmission");
  const locSel = document.getElementById("location");
  const fillBtn = document.getElementById("fill-sample");
  if (!form) return;

  try {
    const res = await fetch("/api/options/car");
    if (res.ok) {
      const data = await res.json();
      companySel.innerHTML = "";
      fuelSel.innerHTML = "";
      if (transSel) transSel.innerHTML = "";
      if (locSel) locSel.innerHTML = "";

      data.companies.forEach((c) => {
        companySel.innerHTML += `<option value="${c}">${c}</option>`;
      });
      data.fuel_types.forEach((f) => {
        fuelSel.innerHTML += `<option value="${f}">${f}</option>`;
      });
      if (transSel && data.transmissions) {
        data.transmissions.forEach((t) => {
          transSel.innerHTML += `<option value="${t}">${t}</option>`;
        });
      }
      if (locSel && data.locations) {
        data.locations.forEach((l) => {
          locSel.innerHTML += `<option value="${l}">${l}</option>`;
        });
      }

      function populateNames(comp) {
        nameSel.innerHTML = "";
        let list = [];
        if (data.models_by_company && data.models_by_company[comp]) {
          list = data.models_by_company[comp];
        } else {
          list = data.car_names.filter((n) =>
            n.toLowerCase().startsWith(comp.toLowerCase())
          );
        }
        if (!list || list.length === 0) list = data.car_names.slice(0, 30);
        list.forEach((n) => {
          nameSel.innerHTML += `<option value="${n}">${n}</option>`;
        });
      }

      companySel.addEventListener("change", () => {
        populateNames(companySel.value);
      });

      companySel.value = CAR_SAMPLE.company;
      populateNames(CAR_SAMPLE.company);
      nameSel.value = CAR_SAMPLE.name;
      fuelSel.value = CAR_SAMPLE.fuel_type;
      if (transSel) transSel.value = CAR_SAMPLE.transmission;
      if (locSel) locSel.value = CAR_SAMPLE.location;
    }
  } catch (e) {
    console.warn("Could not load vehicle options:", e.message);
  }

  if (fillBtn) {
    fillBtn.addEventListener("click", () => {
      companySel.value = CAR_SAMPLE.company;
      companySel.dispatchEvent(new Event("change"));
      nameSel.value = CAR_SAMPLE.name;
      fuelSel.value = CAR_SAMPLE.fuel_type;
      if (transSel) transSel.value = CAR_SAMPLE.transmission;
      if (locSel) locSel.value = CAR_SAMPLE.location;
      document.getElementById("year").value = CAR_SAMPLE.year;
      document.getElementById("slider-year").value = CAR_SAMPLE.year;
      document.getElementById("kms_driven").value = CAR_SAMPLE.kms_driven;
      document.getElementById("slider-kms_driven").value = CAR_SAMPLE.kms_driven;
    });
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideApiError();
    setBtnLoading(submitBtn, true);

    try {
      const body = {
        company: companySel.value,
        name: nameSel.value,
        year: parseInt(document.getElementById("year").value, 10),
        kms_driven: parseInt(document.getElementById("kms_driven").value, 10),
        fuel_type: fuelSel.value,
        transmission: transSel ? transSel.value : undefined,
        location: locSel ? locSel.value : undefined
      };

      const data = await postJSON("/api/predict/car-price", body);
      const pred = data.prediction;

      document.getElementById("result-value").textContent = pred.price_formatted;
      document.getElementById("result-sublabel").textContent = "Estimated Fair Market Valuation (Indian Marketplace Benchmark)";

      renderValueBar("result-chart", "Valuation (INR)", pred.price, 50000, 3500000, "rgba(22, 163, 74, 0.8)");

      if (pred.attributions) {
        renderAttributions("attribution-chart", pred.attributions);
      }

      revealResultPanel();
    } catch (err) {
      showApiError(err.message);
    } finally {
      setBtnLoading(submitBtn, false);
    }
  });
}

// ─── Model 04: IPL Score ─────────────────────────────────────────────────────

const IPL_SAMPLE = {
  batting_team: "Chennai Super Kings",
  bowling_team: "Mumbai Indians",
  runs: 85,
  wickets: 2,
  overs: 10.2,
  runs_last_5: 46,
  wickets_last_5: 1
};

async function initIPL() {
  const form = document.getElementById("ipl-form");
  const submitBtn = document.getElementById("ipl-submit");
  const batSel = document.getElementById("batting_team");
  const bowlSel = document.getElementById("bowling_team");
  const fillBtn = document.getElementById("fill-sample");
  if (!form) return;

  try {
    const res = await fetch("/api/options/ipl");
    if (res.ok) {
      const data = await res.json();
      batSel.innerHTML = "";
      bowlSel.innerHTML = "";
      data.teams.forEach((t) => {
        batSel.innerHTML += `<option value="${t}">${t}</option>`;
        bowlSel.innerHTML += `<option value="${t}">${t}</option>`;
      });
      if (bowlSel.options.length > 1) bowlSel.selectedIndex = 1;
    }
  } catch (e) {
    console.warn("Could not load IPL team options:", e.message);
  }

  if (fillBtn) {
    fillBtn.addEventListener("click", () => {
      batSel.value = IPL_SAMPLE.batting_team;
      bowlSel.value = IPL_SAMPLE.bowling_team;
      document.getElementById("runs").value = IPL_SAMPLE.runs;
      document.getElementById("slider-runs").value = IPL_SAMPLE.runs;
      document.getElementById("wickets").value = IPL_SAMPLE.wickets;
      document.getElementById("slider-wickets").value = IPL_SAMPLE.wickets;
      document.getElementById("overs").value = IPL_SAMPLE.overs;
      document.getElementById("slider-overs").value = IPL_SAMPLE.overs;
      document.getElementById("runs_last_5").value = IPL_SAMPLE.runs_last_5;
      document.getElementById("slider-runs_last_5").value = IPL_SAMPLE.runs_last_5;
      document.getElementById("wickets_last_5").value = IPL_SAMPLE.wickets_last_5;
      document.getElementById("slider-wickets_last_5").value = IPL_SAMPLE.wickets_last_5;
    });
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const batting = batSel.value;
    const bowling = bowlSel.value;
    if (batting === bowling) {
      showApiError("Batting and bowling teams must be distinct.");
      return;
    }

    hideApiError();
    setBtnLoading(submitBtn, true);

    try {
      const body = {
        batting_team: batting,
        bowling_team: bowling,
        runs: parseInt(document.getElementById("runs").value, 10),
        wickets: parseInt(document.getElementById("wickets").value, 10),
        overs: parseFloat(document.getElementById("overs").value),
        runs_last_5: parseInt(document.getElementById("runs_last_5").value, 10),
        wickets_last_5: parseInt(document.getElementById("wickets_last_5").value, 10)
      };

      const data = await postJSON("/api/predict/ipl-score", body);
      const pred = data.prediction;

      document.getElementById("result-value").textContent = `${pred.predicted_score} Runs`;
      document.getElementById("result-sublabel").textContent = "Projected 1st Innings Total (20 Overs)";

      renderValueBar("result-chart", "Projected Runs", pred.predicted_score, 0, 260, "rgba(10, 10, 10, 0.85)");

      if (pred.attributions) {
        renderAttributions("attribution-chart", pred.attributions);
      }

      revealResultPanel();
    } catch (err) {
      showApiError(err.message);
    } finally {
      setBtnLoading(submitBtn, false);
    }
  });
}

// ─── Dataset Explorer (datasets.html) ─────────────────────────────────────────

async function loadDatasetPreview(modelId) {
  const container = document.getElementById("dataset-table-container");
  const statsWrap = document.getElementById("dataset-summary-stats");
  if (!container) return;

  container.innerHTML = `<div style="padding:40px;text-align:center;color:var(--text-muted);">Loading dataset preview records...</div>`;

  try {
    const res = await fetch(`/api/dataset/${modelId}`);
    if (!res.ok) throw new Error("Dataset could not be loaded");
    const data = await res.json();

    if (statsWrap) {
      statsWrap.innerHTML = `
        <div>
          <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;color:var(--text-dim)">Dataset Title</div>
          <div style="font-size:1.05rem;font-weight:700;margin-top:2px">${data.title}</div>
          <div style="font-size:0.78rem;color:var(--text-muted)">${data.domain}</div>
        </div>
        <div>
          <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;color:var(--text-dim)">Total Records</div>
          <div style="font-size:1.05rem;font-weight:700;margin-top:2px">${data.total_rows.toLocaleString()}</div>
          <div style="font-size:0.78rem;color:var(--text-muted)">Verified samples</div>
        </div>
        <div>
          <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;color:var(--text-dim)">Schema Width</div>
          <div style="font-size:1.05rem;font-weight:700;margin-top:2px">${data.columns.length} Features</div>
          <div style="font-size:0.78rem;color:var(--text-muted)">Preprocessed columns</div>
        </div>
        <div>
          <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;color:var(--text-dim)">Source File</div>
          <div style="font-size:0.95rem;font-weight:600;font-family:var(--mono);margin-top:2px">${data.filename}</div>
          <div style="font-size:0.78rem;color:var(--text-muted)">CSV format</div>
        </div>
      `;
    }

    const cols = data.columns.map((c) => c.name);
    let tableHtml = `<table class="dataset-table" id="live-dataset-table"><thead><tr>`;
    cols.forEach((col) => {
      tableHtml += `<th>${col}</th>`;
    });
    tableHtml += `</tr></thead><tbody>`;

    data.sample.forEach((row) => {
      tableHtml += `<tr>`;
      cols.forEach((col) => {
        tableHtml += `<td>${row[col] !== undefined ? row[col] : ""}</td>`;
      });
      tableHtml += `</tr>`;
    });
    tableHtml += `</tbody></table>`;

    container.innerHTML = tableHtml;

    const searchInput = document.getElementById("dataset-search-input");
    if (searchInput) {
      searchInput.addEventListener("input", (e) => {
        const q = e.target.value.toLowerCase();
        const rows = container.querySelectorAll("tbody tr");
        rows.forEach((r) => {
          r.style.display = r.textContent.toLowerCase().includes(q) ? "" : "none";
        });
      });
    }
  } catch (e) {
    container.innerHTML = `<div style="padding:32px;color:var(--danger);">Error loading dataset: ${e.message}</div>`;
  }
}

function initDatasetPage() {
  const selector = document.getElementById("dataset-model-select");
  if (!selector) return;

  selector.addEventListener("change", (e) => {
    loadDatasetPreview(e.target.value);
  });
  loadDatasetPreview(selector.value);
}

// ─── Code Snippets Drawer (docs.html) ─────────────────────────────────────────

function initCodeSnippets() {
  document.querySelectorAll(".code-window").forEach((win) => {
    const tabs = win.querySelectorAll(".code-tab-btn");
    const blocks = win.querySelectorAll(".code-content-block");
    const copyBtn = win.querySelector(".code-copy-btn");

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const lang = tab.dataset.lang;
        tabs.forEach((t) => t.classList.toggle("active", t === tab));
        blocks.forEach((b) => (b.style.display = b.dataset.lang === lang ? "block" : "none"));
      });
    });

    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        const visibleBlock = Array.from(blocks).find((b) => b.style.display !== "none");
        if (visibleBlock) {
          navigator.clipboard.writeText(visibleBlock.textContent.trim());
          const original = copyBtn.textContent;
          copyBtn.textContent = "Copied";
          setTimeout(() => (copyBtn.textContent = original), 2000);
        }
      });
    }
  });
}

// ─── DOM Ready Initialization ────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initMobileNav();
  initRangeSync();
  initDiabetes();
  initBreastCancer();
  initCarPrice();
  initIPL();
  initDatasetPage();
  initCodeSnippets();
});
