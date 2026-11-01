/**
 * Day 12: Axlero Agri-Tech Dashboard Logic
 * Interactivity: Live Prediction, Reactive Counterfactual Simulator, Batch Analytics, and Model Monitoring
 */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initFormInputs();
  initSingleFarmForecast();
  initSimulator();
  initBatchAnalytics();
  initDriftMonitoring();
  initDemoButton();
});

// =========================================================================
// 1. Tab Navigation
// =========================================================================
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  const panels = document.querySelectorAll(".tab-panel");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetId = tab.getAttribute("data-tab");

      tabs.forEach(t => t.classList.remove("active"));
      panels.forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add("active");
      }
    });
  });
}

// =========================================================================
// 2. Form Inputs & Sliders Sync
// =========================================================================
function initFormInputs() {
  // Sync Sliders with Value Display
  const sliders = [
    { id: "nitrogen_kg_ha", valId: "val-nitrogen", unit: " kg/ha" },
    { id: "phosphorus_kg_ha", valId: "val-phosphorus", unit: " kg/ha" },
    { id: "potassium_kg_ha", valId: "val-potassium", unit: " kg/ha" },
  ];

  sliders.forEach(s => {
    const el = document.getElementById(s.id);
    const disp = document.getElementById(s.valId);
    if (el && disp) {
      el.addEventListener("input", (e) => {
        disp.textContent = parseFloat(e.target.value).toFixed(1);
      });
    }
  });

  // Soil pH Slider & Badge
  const phSlider = document.getElementById("soil_ph");
  const phBadge = document.getElementById("ph-badge");
  if (phSlider && phBadge) {
    phSlider.addEventListener("input", (e) => {
      const ph = parseFloat(e.target.value);
      let label = "Optimal";
      let cls = "badge-success";

      if (ph < 5.8) {
        label = "Acidic";
        cls = "badge-warning";
      } else if (ph > 7.5) {
        label = "Alkaline";
        cls = "badge-info";
      }

      phBadge.textContent = `${ph.toFixed(2)} (${label})`;
      phBadge.className = `badge ${cls}`;
    });
  }

  // Crop Selector Cards
  const cropCards = document.querySelectorAll(".crop-card");
  cropCards.forEach(card => {
    card.addEventListener("click", () => {
      cropCards.forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
      const radio = card.querySelector('input[type="radio"]');
      if (radio) radio.checked = true;
    });
  });

  // Reset Button
  const btnReset = document.getElementById("btn-reset-form");
  if (btnReset) {
    btnReset.addEventListener("click", () => {
      document.getElementById("farm-forecast-form").reset();
      document.querySelector('.crop-card[data-crop="Wheat"]').click();
      document.getElementById("val-nitrogen").textContent = "115.0";
      document.getElementById("val-phosphorus").textContent = "48.0";
      document.getElementById("val-potassium").textContent = "55.0";
      document.getElementById("soil_ph").dispatchEvent(new Event("input"));
    });
  }
}

// =========================================================================
// 3. Single Farm Yield Forecast
// =========================================================================
function initSingleFarmForecast() {
  const form = document.getElementById("farm-forecast-form");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = document.getElementById("btn-submit-forecast");
    const originalText = btn.innerHTML;
    btn.innerHTML = `<span class="status-dot"></span> Computing Spatial Inference...`;
    btn.disabled = true;

    const formData = new FormData(form);
    const payload = {
      crop_type: formData.get("crop_type") || "Wheat",
      district_id: formData.get("district_id") || "DIST_101",
      soil_type: formData.get("soil_type") || "Alluvial",
      planted_area_ha: parseFloat(formData.get("planted_area_ha")) || 65.0,
      nitrogen_kg_ha: parseFloat(formData.get("nitrogen_kg_ha")) || 115.0,
      phosphorus_kg_ha: parseFloat(formData.get("phosphorus_kg_ha")) || 48.0,
      potassium_kg_ha: parseFloat(formData.get("potassium_kg_ha")) || 55.0,
      soil_ph: parseFloat(formData.get("soil_ph")) || 5.4,
      organic_matter_pct: parseFloat(formData.get("organic_matter_pct")) || 1.6,
      electrical_conductivity_ds_m: parseFloat(formData.get("electrical_conductivity_ds_m")) || 0.45,
      cumulative_rainfall_mm: parseFloat(formData.get("cumulative_rainfall_mm")) || 820.0,
      mean_temp_max: parseFloat(formData.get("mean_temp_max")) || 33.2,
      mean_vpd_kpa: parseFloat(formData.get("mean_vpd_kpa")) || 1.31,
    };

    try {
      // Attempt live POST to API
      const res = await fetch("/api/v1/prescribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        const data = await res.json();
        updateResultsUI(data);
      } else {
        // Fallback simulation if offline
        const fallback = simulateClientSideInference(payload);
        updateResultsUI(fallback);
      }
    } catch (err) {
      console.warn("API request failed, rendering client-side simulation:", err);
      const fallback = simulateClientSideInference(payload);
      updateResultsUI(fallback);
    } finally {
      btn.innerHTML = originalText;
      btn.disabled = false;
    }
  });
}

function updateResultsUI(data) {
  // Primary KPI
  document.getElementById("kpi-crop-badge").textContent = `${data.crop_type} Spatial Forecast`;
  document.getElementById("kpi-predicted-yield").textContent = data.predicted_yield_mt_ha.toFixed(2);
  
  const ci = data.confidence_interval_95;
  document.getElementById("kpi-lower-bound").textContent = ci.lower_bound_mt_ha.toFixed(2);
  document.getElementById("kpi-upper-bound").textContent = ci.upper_bound_mt_ha.toFixed(2);
  document.getElementById("kpi-ci-spread").textContent = `95% Confidence Interval: ±${ci.margin_of_error_mt_ha.toFixed(2)} MT/Ha`;

  // Production
  const prod = data.harvest_production_mt;
  document.getElementById("kpi-total-production").textContent = `${prod.expected_mt.toLocaleString()} MT`;
  document.getElementById("kpi-production-range").textContent = `${prod.lower_mt.toLocaleString()} – ${prod.upper_mt.toLocaleString()} MT`;

  // Risk Badge
  const risk = data.risk_assessment;
  const riskBadge = document.getElementById("kpi-risk-badge");
  riskBadge.textContent = `${risk.level} RISK`;
  riskBadge.className = `risk-badge risk-badge-${risk.level.toLowerCase()}`;
  document.getElementById("kpi-risk-factor-count").textContent = 
    risk.risk_factors.length ? `${risk.risk_factors.length} Trigger(s): ${risk.risk_factors[0]}` : "Optimal Environment";

  // Stress Gauges
  const stress = risk.stress_indices;
  document.getElementById("stress-ph-val").textContent = `${stress.soil_ph.toFixed(2)} pH`;
  document.getElementById("bar-ph").style.width = `${Math.min(100, Math.max(10, (stress.soil_ph / 9.0) * 100))}%`;

  document.getElementById("stress-drought-val").textContent = stress.drought_severity_index.toFixed(3);
  document.getElementById("bar-drought").style.width = `${Math.min(100, stress.drought_severity_index * 1200)}%`;

  document.getElementById("stress-heat-val").textContent = `${stress.heat_stress_days.toFixed(0)} Days`;
  document.getElementById("bar-heat").style.width = `${Math.min(100, stress.heat_stress_days * 8)}%`;

  document.getElementById("stress-npk-val").textContent = stress.npk_imbalance_index.toFixed(3);
  document.getElementById("bar-npk").style.width = `${Math.min(100, stress.npk_imbalance_index * 300)}%`;

  // Prescriptions
  const rxContainer = document.getElementById("prescriptions-list-container");
  const upsideBadge = document.getElementById("kpi-upside-badge");
  rxContainer.innerHTML = "";

  const prescriptions = data.prescriptive_advisory.prescriptions || [];
  upsideBadge.textContent = `+${data.prescriptive_advisory.max_potential_yield_gain_pct.toFixed(1)}% Yield Upside`;

  if (prescriptions.length === 0) {
    rxContainer.innerHTML = `
      <div class="prescription-card priority-medium">
        <h4 class="rx-title">Optimal Baseline Maintenance</h4>
        <p class="rx-dosage">Maintain current nutrient application rates and standard soil mulch practices.</p>
      </div>`;
  } else {
    prescriptions.forEach(p => {
      const pClass = p.priority === "HIGH" ? "priority-high" : "priority-medium";
      const tagClass = p.priority === "HIGH" ? "tag-high" : "tag-medium";
      const card = document.createElement("div");
      card.className = `prescription-card ${pClass}`;
      card.innerHTML = `
        <div class="rx-header">
          <span class="rx-tag ${tagClass}">${p.priority} PRIORITY</span>
          <span class="rx-gain">+${p.expected_yield_gain_mt_ha.toFixed(2)} MT/Ha (+${p.expected_gain_pct.toFixed(1)}%)</span>
        </div>
        <h4 class="rx-title">${p.action}</h4>
        <p class="rx-dosage"><strong>Dosage:</strong> ${p.dosage}</p>
        <p class="rx-target"><strong>Target:</strong> ${p.target_metric}</p>
      `;
      rxContainer.appendChild(card);
    });
  }

  // Update simulator baseline
  document.getElementById("sim-base-yield").textContent = `${data.predicted_yield_mt_ha.toFixed(2)} MT/Ha`;
  document.getElementById("sim-base-prod").textContent = `${prod.expected_mt.toFixed(1)} MT Total`;
  updateSimulatorOutcomes();
}

// Client-side fallback simulation logic
function simulateClientSideInference(p) {
  let baseYield = 4.2;
  if (p.crop_type === "Rice") baseYield = 4.1;
  else if (p.crop_type === "Maize") baseYield = 5.6;
  else if (p.crop_type === "Soybean") baseYield = 2.7;
  else if (p.crop_type === "Cotton") baseYield = 2.2;

  // pH modifier
  let phFactor = 1.0;
  if (p.soil_ph < 5.8) phFactor -= (5.8 - p.soil_ph) * 0.15;
  if (p.soil_ph > 7.6) phFactor -= (p.soil_ph - 7.6) * 0.12;

  // Nutrient modifier
  let nFactor = Math.min(1.05, p.nitrogen_kg_ha / 115.0);
  const predYield = Number((baseYield * phFactor * nFactor).toFixed(3));
  const stdErr = 0.35;
  const area = p.planted_area_ha;

  const prescriptions = [];
  if (p.soil_ph < 6.0) {
    prescriptions.push({
      action: "Apply Agricultural Limestone (CaCO3)",
      priority: "HIGH",
      dosage: `${((6.2 - p.soil_ph) * 2.0).toFixed(1)} tons/ha applied 4-6 weeks prior to sowing`,
      target_metric: `Elevate soil pH from ${p.soil_ph.toFixed(1)} to 6.5`,
      expected_yield_gain_mt_ha: 0.28,
      expected_gain_pct: (0.28 / predYield) * 100
    });
  }
  if (p.nitrogen_kg_ha < 105.0) {
    prescriptions.push({
      action: "Synchronized Nitrogen Top-Dressing",
      priority: "HIGH",
      dosage: `Top-dress with ${(120.0 - p.nitrogen_kg_ha).toFixed(0)} kg/ha nitrogen split across tillering and heading`,
      target_metric: `Restore Nitrogen from ${p.nitrogen_kg_ha.toFixed(1)} to 120.0 kg/ha`,
      expected_yield_gain_mt_ha: 0.22,
      expected_gain_pct: (0.22 / predYield) * 100
    });
  }
  if (p.organic_matter_pct < 2.0) {
    prescriptions.push({
      action: "Biochar & Green Manure Incorporation",
      priority: "MEDIUM",
      dosage: "Incorporate 6-8 MT/ha cover crop biomass or vermicompost",
      target_metric: `Elevate organic matter from ${p.organic_matter_pct.toFixed(1)}% to 2.5%`,
      expected_yield_gain_mt_ha: 0.15,
      expected_gain_pct: (0.15 / predYield) * 100
    });
  }

  const maxGain = prescriptions.length ? Math.max(...prescriptions.map(x => x.expected_gain_pct)) : 0.0;

  return {
    crop_type: p.crop_type,
    planted_area_ha: area,
    predicted_yield_mt_ha: predYield,
    uncertainty_std_mt_ha: stdErr,
    confidence_interval_95: {
      lower_bound_mt_ha: Math.max(0.5, predYield - 1.96 * stdErr),
      upper_bound_mt_ha: predYield + 1.96 * stdErr,
      margin_of_error_mt_ha: 1.96 * stdErr
    },
    harvest_production_mt: {
      expected_mt: predYield * area,
      lower_mt: Math.max(0.5, predYield - 1.96 * stdErr) * area,
      upper_mt: (predYield + 1.96 * stdErr) * area
    },
    risk_assessment: {
      level: p.soil_ph < 5.8 ? "MODERATE" : "LOW",
      risk_factors: p.soil_ph < 5.8 ? [`Soil Acidity Stress (pH ${p.soil_ph.toFixed(2)} < 5.8)`] : [],
      stress_indices: {
        soil_ph: p.soil_ph,
        drought_severity_index: 0.0325,
        heat_stress_days: 5.0,
        npk_imbalance_index: 0.085
      }
    },
    prescriptive_advisory: {
      max_potential_yield_gain_pct: maxGain,
      prescriptions: prescriptions
    }
  };
}

// =========================================================================
// 4. Interactive What-If Counterfactual Simulator
// =========================================================================
function initSimulator() {
  const simControls = [
    { id: "sim-slider-ph", valId: "sim-val-ph", prefix: "" },
    { id: "sim-slider-n", valId: "sim-val-n", prefix: "+" },
    { id: "sim-slider-p", valId: "sim-val-p", prefix: "+" },
    { id: "sim-slider-som", valId: "sim-val-som", prefix: "" }
  ];

  simControls.forEach(c => {
    const el = document.getElementById(c.id);
    const disp = document.getElementById(c.valId);
    if (el && disp) {
      el.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        disp.textContent = `${c.prefix}${val.toFixed(c.id.includes("som") ? 1 : (c.id.includes("ph") ? 2 : 0))}`;
        updateSimulatorOutcomes();
      });
    }
  });

  updateSimulatorOutcomes();
}

function updateSimulatorOutcomes() {
  const baseYieldEl = document.getElementById("sim-base-yield");
  const baseYield = parseFloat(baseYieldEl.textContent) || 3.92;

  const simPh = parseFloat(document.getElementById("sim-slider-ph").value) || 6.5;
  const simN = parseFloat(document.getElementById("sim-slider-n").value) || 25.0;
  const simP = parseFloat(document.getElementById("sim-slider-p").value) || 15.0;
  const simSOM = parseFloat(document.getElementById("sim-slider-som").value) || 2.4;

  // Agronomic Response Formula
  const phGain = Math.max(0, (simPh - 5.4) * 0.22);
  const nGain = (simN / 50.0) * 0.18;
  const pGain = (simP / 40.0) * 0.09;
  const somGain = (simSOM - 1.6) * 0.12;

  const totalGain = Math.max(0.05, phGain + nGain + pGain + somGain);
  const optYield = Number((baseYield + totalGain).toFixed(2));
  const gainPct = ((totalGain / baseYield) * 100).toFixed(1);

  const area = 65.0;
  const baseProd = baseYield * area;
  const optProd = optYield * area;
  const addedProd = (optYield - baseYield) * area;

  // Financial ROI
  const cropPricePerMT = 280; // $ USD
  const addedRevenue = addedProd * cropPricePerMT;
  const inputCost = (simPh > 5.5 ? 900 : 0) + (simN * 12) + (simP * 18) + 400;
  const netProfit = addedRevenue - inputCost;
  const roiPct = Math.round((netProfit / Math.max(1, inputCost)) * 100);

  // Update UI
  document.getElementById("sim-gain-pct").textContent = `+${gainPct}%`;
  document.getElementById("sim-opt-yield").textContent = `${optYield} MT/Ha`;
  document.getElementById("sim-opt-prod").textContent = `${optProd.toFixed(1)} MT Total`;
  document.getElementById("sim-net-increase").textContent = `+${totalGain.toFixed(2)} MT/Ha (+${addedProd.toFixed(1)} MT)`;
  document.getElementById("sim-added-rev").textContent = `+$${Math.round(addedRevenue).toLocaleString()} USD`;
  document.getElementById("sim-input-cost").textContent = `$${Math.round(inputCost).toLocaleString()} USD`;
  document.getElementById("sim-net-roi").textContent = `+${roiPct}% (${(addedRevenue / Math.max(1, inputCost)).toFixed(1)}x Return)`;
}

// =========================================================================
// 5. Regional Cooperative Batch Analytics Table
// =========================================================================
const SAMPLE_BATCH_DATA = [
  { district: "DIST_101", year: 2023, crop: "Wheat", area: 75.0, yield: 4.37, ci_low: 3.52, ci_high: 5.22, prod: 327.8, risk: "HIGH", prescription: "Apply Agricultural Limestone (CaCO3)", upside: "+8.5%" },
  { district: "DIST_101", year: 2023, crop: "Rice", area: 60.0, yield: 4.09, ci_low: 3.28, ci_high: 4.90, prod: 245.4, risk: "HIGH", prescription: "Synchronized Nitrogen Top-Dressing", upside: "+6.8%" },
  { district: "DIST_102", year: 2023, crop: "Maize", area: 85.0, yield: 5.61, ci_low: 4.65, ci_high: 6.57, prod: 476.9, risk: "HIGH", prescription: "Synchronized Nitrogen Top-Dressing", upside: "+4.2%" },
  { district: "DIST_102", year: 2023, crop: "Soybean", area: 45.0, yield: 2.66, ci_low: 2.10, ci_high: 3.22, prod: 119.7, risk: "HIGH", prescription: "Biochar & Green Manure Incorporation", upside: "+7.5%" },
  { district: "DIST_103", year: 2023, crop: "Cotton", area: 90.0, yield: 2.15, ci_low: 1.55, ci_high: 2.75, prod: 193.5, risk: "CRITICAL", prescription: "Apply Agricultural Limestone (CaCO3)", upside: "+11.0%" },
  { district: "DIST_104", year: 2023, crop: "Wheat", area: 68.0, yield: 4.42, ci_low: 3.60, ci_high: 5.24, prod: 300.6, risk: "HIGH", prescription: "Synchronized Nitrogen Top-Dressing", upside: "+5.1%" },
  { district: "DIST_105", year: 2023, crop: "Rice", area: 72.0, yield: 4.15, ci_low: 3.35, ci_high: 4.95, prod: 298.8, risk: "HIGH", prescription: "Rock Phosphate & Mycorrhizal Inoculant", upside: "+4.8%" },
  { district: "DIST_106", year: 2023, crop: "Maize", area: 80.0, yield: 5.58, ci_low: 4.60, ci_high: 6.56, prod: 446.4, risk: "HIGH", prescription: "Biochar & Green Manure Incorporation", upside: "+3.9%" },
  { district: "DIST_107", year: 2023, crop: "Wheat", area: 55.0, yield: 4.29, ci_low: 3.45, ci_high: 5.13, prod: 236.0, risk: "MODERATE", prescription: "Synchronized Nitrogen Top-Dressing", upside: "+4.6%" },
  { district: "DIST_108", year: 2023, crop: "Cotton", area: 65.0, yield: 2.18, ci_low: 1.62, ci_high: 2.74, prod: 141.7, risk: "CRITICAL", prescription: "Apply Agricultural Limestone (CaCO3)", upside: "+9.8%" },
];

function initBatchAnalytics() {
  renderBatchTable(SAMPLE_BATCH_DATA);

  const searchInput = document.getElementById("batch-search");
  const filterCrop = document.getElementById("batch-filter-crop");

  const filterHandler = () => {
    const q = searchInput.value.toLowerCase();
    const crop = filterCrop.value;

    const filtered = SAMPLE_BATCH_DATA.filter(row => {
      const matchSearch = row.district.toLowerCase().includes(q) || row.crop.toLowerCase().includes(q) || row.prescription.toLowerCase().includes(q);
      const matchCrop = crop === "ALL" || row.crop === crop;
      return matchSearch && matchCrop;
    });
    renderBatchTable(filtered);
  };

  if (searchInput) searchInput.addEventListener("input", filterHandler);
  if (filterCrop) filterCrop.addEventListener("change", filterHandler);

  // CSV Export
  const btnExport = document.getElementById("btn-export-batch");
  if (btnExport) {
    btnExport.addEventListener("click", () => {
      let csvContent = "data:text/csv;charset=utf-8,District,Year,Crop,Area_Ha,Yield_MT_Ha,CI_Lower,CI_Upper,Production_MT,Risk,Prescription,Upside\n";
      SAMPLE_BATCH_DATA.forEach(r => {
        csvContent += `${r.district},${r.year},${r.crop},${r.area},${r.yield},${r.ci_low},${r.ci_high},${r.prod},${r.risk},"${r.prescription}",${r.upside}\n`;
      });
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", "regional_crop_yield_batch_predictions.csv");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  }
}

function renderBatchTable(data) {
  const tbody = document.getElementById("batch-table-body");
  if (!tbody) return;
  tbody.innerHTML = "";

  if (data.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; color: var(--text-muted); padding: 20px;">No regional farm records matched the active filter.</td></tr>`;
    return;
  }

  data.forEach(r => {
    const tr = document.createElement("tr");
    const riskClass = r.risk.toLowerCase();
    tr.innerHTML = `
      <td><strong>${r.district}</strong></td>
      <td>${r.year}</td>
      <td><span class="crop-badge">${r.crop}</span></td>
      <td>${r.area}</td>
      <td><strong>${r.yield.toFixed(2)}</strong> MT/Ha</td>
      <td><span class="mono-span">${r.ci_low.toFixed(2)} – ${r.ci_high.toFixed(2)}</span></td>
      <td>${r.prod.toFixed(1)} MT</td>
      <td><span class="risk-badge risk-badge-${riskClass}">${r.risk}</span></td>
      <td>${r.prescription}</td>
      <td><strong class="text-success">${r.upside}</strong></td>
    `;
    tbody.appendChild(tr);
  });
}

// =========================================================================
// 6. Covariate Drift & Monitoring
// =========================================================================
const SAMPLE_DRIFT_FEATURES = [
  { name: "nitrogen_kg_ha", psi: 0.0000, ks: 0.0000, pval: 1.0, base: "112.70 ± 30.26", prod: "112.70 ± 30.26", shift: "+0.0%", status: "STABLE" },
  { name: "phosphorus_kg_ha", psi: 0.0000, ks: 0.0000, pval: 1.0, base: "44.13 ± 15.04", prod: "44.13 ± 15.04", shift: "+0.0%", status: "STABLE" },
  { name: "potassium_kg_ha", psi: 0.0000, ks: 0.0000, pval: 1.0, base: "160.18 ± 47.79", prod: "160.18 ± 47.79", shift: "+0.0%", status: "STABLE" },
  { name: "soil_ph", psi: 0.0000, ks: 0.0000, pval: 1.0, base: "6.32 ± 0.63", prod: "6.32 ± 0.63", shift: "+0.0%", status: "STABLE" },
  { name: "organic_matter_pct", psi: 0.0000, ks: 0.0000, pval: 1.0, base: "1.19 ± 0.28", prod: "1.19 ± 0.28", shift: "+0.0%", status: "STABLE" },
  { name: "cumulative_rainfall_mm", psi: 0.0000, ks: 0.0000, pval: 1.0, base: "884.10 ± 404.02", prod: "884.10 ± 404.02", shift: "+0.0%", status: "STABLE" },
  { name: "mean_temp_max", psi: 0.0000, ks: 0.0000, pval: 1.0, base: "32.47 ± 3.99", prod: "32.47 ± 3.99", shift: "+0.0%", status: "STABLE" },
  { name: "mean_vpd_kpa", psi: 0.0000, ks: 0.0000, pval: 1.0, base: "1.32 ± 0.13", prod: "1.32 ± 0.13", shift: "+0.0%", status: "STABLE" },
  { name: "heat_stress_days", psi: 0.0000, ks: 0.0000, pval: 1.0, base: "72.23 ± 40.97", prod: "72.23 ± 40.97", shift: "+0.0%", status: "STABLE" }
];

function initDriftMonitoring() {
  renderDriftTable(SAMPLE_DRIFT_FEATURES);

  const btnRefresh = document.getElementById("btn-refresh-drift");
  if (btnRefresh) {
    btnRefresh.addEventListener("click", async () => {
      btnRefresh.disabled = true;
      btnRefresh.innerHTML = `<span class="status-dot"></span> Scanning Covariates...`;

      try {
        const res = await fetch("/api/v1/monitoring");
        if (res.ok) {
          const data = await res.json();
          document.getElementById("drift-health-score").textContent = `${data.health_score_pct}%`;
          document.getElementById("drift-cnt-stable").textContent = `${data.stable_feature_count} / ${data.total_monitored_features}`;
          document.getElementById("drift-cnt-moderate").textContent = data.moderate_drift_feature_count;
          document.getElementById("drift-cnt-high").textContent = data.high_drift_feature_count;
          
          const featArray = Object.values(data.features).map(f => ({
            name: f.feature,
            psi: f.psi,
            ks: f.ks_statistic,
            pval: f.ks_p_value,
            base: `${f.baseline_mean} ± ${f.baseline_std}`,
            prod: `${f.current_mean} ± ${f.current_std}`,
            shift: `${f.mean_shift_pct > 0 ? "+" : ""}${f.mean_shift_pct}%`,
            status: f.status
          }));
          renderDriftTable(featArray);
        }
      } catch (e) {
        console.warn("Using offline drift matrix data");
      } finally {
        setTimeout(() => {
          btnRefresh.disabled = false;
          btnRefresh.innerHTML = `
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
            </svg> Run Drift Re-Scan`;
        }, 500);
      }
    });
  }
}

function renderDriftTable(features) {
  const tbody = document.getElementById("drift-table-body");
  if (!tbody) return;
  tbody.innerHTML = "";

  features.forEach(f => {
    const tr = document.createElement("tr");
    const statusIcon = f.status === "STABLE" ? "🟢" : (f.status === "MODERATE_DRIFT" ? "🟡" : "🔴");
    tr.innerHTML = `
      <td><code>${f.name}</code></td>
      <td><strong>${f.psi.toFixed(4)}</strong></td>
      <td>${f.ks.toFixed(4)}</td>
      <td>${f.pval.toFixed(3)}</td>
      <td>${f.base}</td>
      <td>${f.prod}</td>
      <td><span class="${f.shift.includes('+') ? 'text-success' : 'text-warning'}">${f.shift}</span></td>
      <td>${statusIcon} <strong>${f.status}</strong></td>
    `;
    tbody.appendChild(tr);
  });
}

// =========================================================================
// 7. Quick Demo Loader Button
// =========================================================================
function initDemoButton() {
  const btn = document.getElementById("btn-quick-demo");
  if (!btn) return;

  btn.addEventListener("click", () => {
    // Populate an acidic farm scenario needing limestone amendment
    document.querySelector('.crop-card[data-crop="Wheat"]').click();
    document.getElementById("district_id").value = "DIST_101";
    document.getElementById("soil_type").value = "Alluvial";
    document.getElementById("planted_area_ha").value = 75.0;

    const n = document.getElementById("nitrogen_kg_ha");
    n.value = 95.0;
    n.dispatchEvent(new Event("input"));

    const p = document.getElementById("phosphorus_kg_ha");
    p.value = 35.0;
    p.dispatchEvent(new Event("input"));

    const k = document.getElementById("potassium_kg_ha");
    k.value = 40.0;
    k.dispatchEvent(new Event("input"));

    const ph = document.getElementById("soil_ph");
    ph.value = 5.35;
    ph.dispatchEvent(new Event("input"));

    document.getElementById("organic_matter_pct").value = 1.4;
    document.getElementById("electrical_conductivity_ds_m").value = 0.40;
    document.getElementById("cumulative_rainfall_mm").value = 850.0;
    document.getElementById("mean_temp_max").value = 33.4;
    document.getElementById("mean_vpd_kpa").value = 1.35;

    // Trigger submission
    document.getElementById("farm-forecast-form").dispatchEvent(new Event("submit"));
  });
}
