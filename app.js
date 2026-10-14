/**
 * DDPI Analytics Dashboard • Client-Side Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  initDashboard();
});

let statusPollTimer = null;

function initDashboard() {
  setupTabs();
  setupModal();
  setupPredictor();
  setupPipelineRunner();
  loadResults();
}

/**
 * Fetch and render results from results.json via /api/results
 */
async function loadResults() {
  try {
    const res = await fetch("/api/results");
    if (!res.ok) return;
    const data = await res.json();
    if (data.error) {
      console.log(data.error);
      return;
    }

    // 1. Update KPIs
    if (data.dataset_shape) {
      document.getElementById("kpi-rows").textContent = data.dataset_shape.rows.toLocaleString();
      document.getElementById("kpi-cols").textContent = `${data.dataset_shape.columns} Features Collected`;
    }
    if (data.detected_likert_count) {
      document.getElementById("kpi-likert").textContent = data.detected_likert_count;
    }
    if (data.best_model) {
      document.getElementById("kpi-best-model").textContent = data.best_model;
    }
    if (data.best_auc) {
      document.getElementById("kpi-auc").textContent = data.best_auc.toFixed(3);
    }

    // 2. Update Model Comparison Table
    if (data.model_comparison && Array.isArray(data.model_comparison)) {
      const tbody = document.getElementById("models-table-body");
      tbody.innerHTML = "";
      data.model_comparison.forEach((m, idx) => {
        const isBest = idx === 0;
        if (isBest) {
          document.getElementById("kpi-mae").textContent = m.MAE.toFixed(4);
          document.getElementById("kpi-r2").textContent = m.R2.toFixed(4);
        }
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${m.Model}</strong></td>
          <td><span class="badge-accent">${m.MAE.toFixed(4)}</span></td>
          <td>${m.RMSE.toFixed(4)}</td>
          <td>${m.R2.toFixed(4)}</td>
          <td>${isBest ? '<span class="pill-best">Best Fit</span>' : '<span style="color:var(--text-muted)">Evaluated</span>'}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    // 3. Update Cluster Counts
    if (data.cluster_counts) {
      const container = document.getElementById("cluster-bars-container");
      container.innerHTML = "";
      const clusterNames = {
        "0": "Creative Explorers",
        "1": "Reflective Thinkers",
        "2": "Conscientious Achievers",
        "3": "Social Connectors"
      };
      const colors = ["fill-purple", "fill-emerald", "fill-amber", "fill-cyan"];
      const total = Object.values(data.cluster_counts).reduce((a, b) => a + b, 0);

      Object.entries(data.cluster_counts).forEach(([cid, count], i) => {
        const pct = ((count / total) * 100).toFixed(1);
        const colorClass = colors[i % colors.length];
        const row = document.createElement("div");
        row.className = "cluster-row";
        row.innerHTML = `
          <span class="c-label">Cluster ${cid} (${clusterNames[cid] || "Archetype"})</span>
          <div class="c-bar-track">
            <div class="c-bar-fill ${colorClass}" style="width: ${pct}%;"></div>
          </div>
          <span class="c-count">${count}</span>
        `;
        container.appendChild(row);
      });
    }

    // 4. Update Gender Trait Differences
    if (data.mean_traits_by_gender) {
      const container = document.getElementById("gender-traits-container");
      container.innerHTML = "";
      const f = data.mean_traits_by_gender.female || {};
      const m = data.mean_traits_by_gender.male || {};
      const traits = ["Conscientiousness", "Agreeableness", "Openness", "Emotional_Stability", "Extraversion"];

      traits.forEach(t => {
        if (f[t] !== undefined && m[t] !== undefined) {
          const row = document.createElement("div");
          row.className = "trait-stat-row";
          const fStr = f[t] >= 0 ? `+${f[t].toFixed(2)}` : f[t].toFixed(2);
          const mStr = m[t] >= 0 ? `+${m[t].toFixed(2)}` : m[t].toFixed(2);
          row.innerHTML = `
            <span class="t-name">${t.replace('_', ' ')}</span>
            <span class="t-val female-color">Female: ${fStr}</span>
            <span class="t-val male-color">Male: ${mStr}</span>
          `;
          container.appendChild(row);
        }
      });
    }

    // Refresh image caches with timestamp
    refreshChartImages();
  } catch (err) {
    console.error("Failed to load results:", err);
  }
}

function refreshChartImages() {
  const ts = Date.now();
  ["img-target-dist", "img-corr-heatmap", "img-model-mae", "img-roc-curve"].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      const baseSrc = el.src.split("?")[0];
      el.src = `${baseSrc}?t=${ts}`;
    }
  });
}

/**
 * Pipeline Execution and Console Log Polling
 */
function setupPipelineRunner() {
  const runBtn = document.getElementById("run-pipeline-btn");
  const clearBtn = document.getElementById("clear-terminal-btn");
  const terminalLog = document.getElementById("terminal-log");
  const liveIndicator = document.getElementById("live-indicator");
  const statusBadge = document.getElementById("system-status-text");

  clearBtn.addEventListener("click", () => {
    terminalLog.textContent = "";
  });

  runBtn.addEventListener("click", async () => {
    if (runBtn.classList.contains("loading")) return;

    runBtn.classList.add("loading");
    document.getElementById("run-btn-label").textContent = "Running Pipeline...";
    liveIndicator.textContent = "RUNNING";
    liveIndicator.classList.add("running");
    statusBadge.textContent = "Processing";

    terminalLog.textContent = "Initiating Python execution: python -u DDPI.py...\n";

    try {
      const res = await fetch("/api/run", { method: "POST" });
      if (!res.ok) {
        const err = await res.json();
        terminalLog.textContent += `\nError: ${err.error || "Execution failed to start."}`;
        resetRunState();
        return;
      }
      pollPipelineStatus();
    } catch (err) {
      terminalLog.textContent += `\nNetwork error: ${err.message}`;
      resetRunState();
    }
  });

  function pollPipelineStatus() {
    if (statusPollTimer) clearInterval(statusPollTimer);

    statusPollTimer = setInterval(async () => {
      try {
        const res = await fetch("/api/status");
        if (!res.ok) return;
        const status = await res.json();

        if (status.last_log) {
          terminalLog.textContent = status.last_log;
          const body = document.getElementById("terminal-body");
          body.scrollTop = body.scrollHeight;
        }

        if (!status.is_running) {
          clearInterval(statusPollTimer);
          statusPollTimer = null;
          resetRunState();
          loadResults();
        }
      } catch (err) {
        console.error("Poll status error:", err);
      }
    }, 750);
  }

  function resetRunState() {
    runBtn.classList.remove("loading");
    document.getElementById("run-btn-label").textContent = "Run Analysis";
    liveIndicator.textContent = "IDLE";
    liveIndicator.classList.remove("running");
    statusBadge.textContent = "Ready";
  }
}

/**
 * Interactive Disposition Predictor with Debounce
 */
function setupPredictor() {
  const sliders = [
    "slider-openness",
    "slider-conscientiousness",
    "slider-extraversion",
    "slider-agreeableness",
    "slider-emotional-stability"
  ];

  sliders.forEach(id => {
    const el = document.getElementById(id);
    const valDisplay = document.getElementById(id.replace("slider-", "val-"));
    el.addEventListener("input", () => {
      const num = parseFloat(el.value);
      valDisplay.textContent = (num >= 0 ? "+" : "") + num.toFixed(2);
      debouncedPredict();
    });
  });

  ["select-gender", "input-age", "input-bmi"].forEach(id => {
    document.getElementById(id).addEventListener("change", debouncedPredict);
  });

  let debounceTimeout = null;
  function debouncedPredict() {
    if (debounceTimeout) clearTimeout(debounceTimeout);
    debounceTimeout = setTimeout(sendPredictionRequest, 150);
  }

  async function sendPredictionRequest() {
    const payload = {
      Openness: parseFloat(document.getElementById("slider-openness").value),
      Conscientiousness: parseFloat(document.getElementById("slider-conscientiousness").value),
      Extraversion: parseFloat(document.getElementById("slider-extraversion").value),
      Agreeableness: parseFloat(document.getElementById("slider-agreeableness").value),
      Emotional_Stability: parseFloat(document.getElementById("slider-emotional-stability").value),
      Gender: document.getElementById("select-gender").value,
      Age: parseFloat(document.getElementById("input-age").value) || 23,
      BMI: parseFloat(document.getElementById("input-bmi").value) || 21.8
    };

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) return;
      const data = await res.json();

      // Update UI elements
      document.getElementById("res-happiness-score").textContent = data.predicted_happiness.toFixed(2);
      document.getElementById("res-tier-badge").textContent = data.happiness_tier;
      
      const pct = Math.min(100, Math.max(10, (data.predicted_happiness / 5.0) * 100));
      document.getElementById("res-gauge-fill").style.width = `${pct}%`;

      if (data.cluster) {
        document.getElementById("res-cluster-name").textContent = data.cluster.name;
        document.getElementById("res-cluster-desc").textContent = data.cluster.desc;
      }

      if (data.recommended_interests && Array.isArray(data.recommended_interests)) {
        const container = document.getElementById("res-interest-tags");
        container.innerHTML = "";
        data.recommended_interests.forEach(interest => {
          const span = document.createElement("span");
          span.className = "tag-pill";
          span.textContent = interest;
          container.appendChild(span);
        });
      }
    } catch (err) {
      console.error("Prediction error:", err);
    }
  }

  // Run initial prediction once
  sendPredictionRequest();
}

/**
 * Gallery Tabs
 */
function setupTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(btn.dataset.tab);
      if (targetPane) {
        targetPane.classList.add("active");
      }
    });
  });
}

/**
 * Lightbox Modal for Fullscreen Chart Inspection
 */
function setupModal() {
  const modal = document.getElementById("image-modal");
  const modalImg = document.getElementById("modal-img");
  const closeBtn = document.getElementById("modal-close-btn");

  document.querySelectorAll(".image-wrapper img").forEach(img => {
    img.addEventListener("click", () => {
      modalImg.src = img.src;
      modal.classList.add("open");
    });
  });

  const closeModal = () => modal.classList.remove("open");
  closeBtn.addEventListener("click", closeModal);
  modal.addEventListener("click", e => {
    if (e.target === modal) closeModal();
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && modal.classList.contains("open")) closeModal();
  });
}
