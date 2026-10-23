// ==========================================================================
// GRAPH-BASED RECOMMENDATION SYSTEM - CLIENT CONTROLLER
// High-End Luxury Noir, White & Silver Experience
// ==========================================================================

let networkInstance = null;
let currentUserId = "U0001";
let allUsers = [];
let allProducts = [];
let currentCategoryFilter = "ALL";
let currentGraphMode = "ego"; // "ego", "sample", "explain"
let currentExplainProductId = null;

// Currency formatter for Indian Rupees (₹)
function formatINR(val) {
  const num = Number(val || 0);
  return "₹" + num.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// Curated high-resolution Unsplash product photography mapping
const PRODUCT_IMAGE_MAP = {
  // Electronics
  "smartphone": "https://images.unsplash.com/photo-1598327105666-5b89351aff97?auto=format&fit=crop&w=600&q=80",
  "wireless earbuds": "https://images.unsplash.com/photo-1590658268037-6bf12165a8df?auto=format&fit=crop&w=600&q=80",
  "gaming laptop": "https://images.unsplash.com/photo-1603302576837-37561b2e2302?auto=format&fit=crop&w=600&q=80",
  "smartwatch": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?auto=format&fit=crop&w=600&q=80",
  "4k monitor": "https://images.unsplash.com/photo-1527443224154-c4a3942d3acf?auto=format&fit=crop&w=600&q=80",
  "bluetooth speaker": "https://images.unsplash.com/photo-1545454675-3531b543be5d?auto=format&fit=crop&w=600&q=80",
  "tablet pro": "https://images.unsplash.com/photo-1544244015-0df4b3ffc6b0?auto=format&fit=crop&w=600&q=80",
  "smartphone v2": "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?auto=format&fit=crop&w=600&q=80",
  "smartphone #49": "https://images.unsplash.com/photo-1592750475338-74b7b21085ab?auto=format&fit=crop&w=600&q=80",

  // Fashion
  "leather jacket": "https://images.unsplash.com/photo-1551028719-00167b16eac5?auto=format&fit=crop&w=600&q=80",
  "leather jacket v2": "https://images.unsplash.com/photo-1521223890158-f9f7c3d5d504?auto=format&fit=crop&w=600&q=80",
  "running shoes": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&w=600&q=80",
  "denim jeans": "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?auto=format&fit=crop&w=600&q=80",
  "cotton hoodie": "https://images.unsplash.com/photo-1556905055-8f358a7a47b2?auto=format&fit=crop&w=600&q=80",
  "designer sunglasses": "https://images.unsplash.com/photo-1511499767150-a48a237f0083?auto=format&fit=crop&w=600&q=80",
  "wristwatch": "https://images.unsplash.com/photo-1524805444758-089113d48a6d?auto=format&fit=crop&w=600&q=80",
  "sneakers": "https://images.unsplash.com/photo-1560769629-975ec94e6a86?auto=format&fit=crop&w=600&q=80",

  // Home & Kitchen
  "espresso maker": "https://images.unsplash.com/photo-1517668808822-9ebb02f2a0e6?auto=format&fit=crop&w=600&q=80",
  "espresso maker v2": "https://images.unsplash.com/photo-1517668808822-9ebb02f2a0e6?auto=format&fit=crop&w=600&q=80",
  "air fryer": "https://images.unsplash.com/photo-1585515320310-259814833e62?auto=format&fit=crop&w=600&q=80",
  "chef knife set": "https://images.unsplash.com/photo-1593618998160-e34014e67546?auto=format&fit=crop&w=600&q=80",
  "robot vacuum": "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=600&q=80",
  "ergonomic chair": "https://images.unsplash.com/photo-1580481077195-c22ae4534c79?auto=format&fit=crop&w=600&q=80",
  "blender": "https://images.unsplash.com/photo-1570222094114-d054a817e56b?auto=format&fit=crop&w=600&q=80",
  "aromatherapy diffuser": "https://images.unsplash.com/photo-1608571423902-eed4a5ad8108?auto=format&fit=crop&w=600&q=80",

  // Sports & Fitness
  "yoga mat": "https://images.unsplash.com/photo-1601925260368-ae2f83cf8b7f?auto=format&fit=crop&w=600&q=80",
  "yoga mat v2": "https://images.unsplash.com/photo-1601925260368-ae2f83cf8b7f?auto=format&fit=crop&w=600&q=80",
  "adjustable dumbbells": "https://images.unsplash.com/photo-1586401100295-7a8096fd231a?auto=format&fit=crop&w=600&q=80",
  "adjustable dumbbells v2": "https://images.unsplash.com/photo-1586401100295-7a8096fd231a?auto=format&fit=crop&w=600&q=80",
  "resistance bands": "https://images.unsplash.com/photo-1598289431512-b97b0917affc?auto=format&fit=crop&w=600&q=80",
  "hydration flask": "https://images.unsplash.com/photo-1602143407151-7111542de6e8?auto=format&fit=crop&w=600&q=80",
  "treadmill": "https://images.unsplash.com/photo-1576678927484-cc907957088c?auto=format&fit=crop&w=600&q=80",
  "cycling helmet": "https://images.unsplash.com/photo-1559348349-86f1f65817fe?auto=format&fit=crop&w=600&q=80",

  // Books & Media
  "machine learning handbook": "https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?auto=format&fit=crop&w=600&q=80",
  "machine learning handbook v2": "https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?auto=format&fit=crop&w=600&q=80",
  "sci-fi best-seller": "https://images.unsplash.com/photo-1532012164546-f432f2e3777f?auto=format&fit=crop&w=600&q=80",
  "sci-fi best-seller v2": "https://images.unsplash.com/photo-1532012164546-f432f2e3777f?auto=format&fit=crop&w=600&q=80",
  "productivity guide": "https://images.unsplash.com/photo-1497633762265-9d179a990aa6?auto=format&fit=crop&w=600&q=80",
  "productivity guide v2": "https://images.unsplash.com/photo-1497633762265-9d179a990aa6?auto=format&fit=crop&w=600&q=80",
  "cooking masterclass": "https://images.unsplash.com/photo-1556911220-e15b29be8c8f?auto=format&fit=crop&w=600&q=80",
  "historical novel": "https://images.unsplash.com/photo-1495446815901-a7297e633e8d?auto=format&fit=crop&w=600&q=80",

  // Beauty & Wellness
  "hydrating serum": "https://images.unsplash.com/photo-1620916566398-39f1143ab7be?auto=format&fit=crop&w=600&q=80",
  "hydrating serum v2": "https://images.unsplash.com/photo-1620916566398-39f1143ab7be?auto=format&fit=crop&w=600&q=80",
  "sunscreen spf 50": "https://images.unsplash.com/photo-1556228720-195a672e8a03?auto=format&fit=crop&w=600&q=80",
  "sunscreen spf 50 v2": "https://images.unsplash.com/photo-1556228720-195a672e8a03?auto=format&fit=crop&w=600&q=80",
  "sunscreen spf 50 #50": "https://images.unsplash.com/photo-1556228720-195a672e8a03?auto=format&fit=crop&w=600&q=80",
  "electric toothbrush": "https://images.unsplash.com/photo-1559591937-e1032a26fa3a?auto=format&fit=crop&w=600&q=80",
  "electric toothbrush v2": "https://images.unsplash.com/photo-1559591937-e1032a26fa3a?auto=format&fit=crop&w=600&q=80",
  "hair dryer": "https://images.unsplash.com/photo-1522337360788-8b13dee7a37e?auto=format&fit=crop&w=600&q=80",
  "organic face wash": "https://images.unsplash.com/photo-1556228722-d0b5de70b774?auto=format&fit=crop&w=600&q=80"
};

// Generic fallback images per category
const CATEGORY_IMAGE_MAP = {
  "electronics": "https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&w=600&q=80",
  "fashion": "https://images.unsplash.com/photo-1490481651871-ab68de25d43d?auto=format&fit=crop&w=600&q=80",
  "home & kitchen": "https://images.unsplash.com/photo-1556911220-e15b29be8c8f?auto=format&fit=crop&w=600&q=80",
  "sports & fitness": "https://images.unsplash.com/photo-1517838277536-f5f99be501cd?auto=format&fit=crop&w=600&q=80",
  "books & media": "https://images.unsplash.com/photo-1497633762265-9d179a990aa6?auto=format&fit=crop&w=600&q=80",
  "beauty & wellness": "https://images.unsplash.com/photo-1522337360788-8b13dee7a37e?auto=format&fit=crop&w=600&q=80"
};

function getProductImageUrl(product) {
  if (!product) return CATEGORY_IMAGE_MAP["electronics"];
  const nameKey = (product.name || "").toLowerCase().trim();
  if (PRODUCT_IMAGE_MAP[nameKey]) {
    return PRODUCT_IMAGE_MAP[nameKey];
  }
  const catKey = (product.category || "").toLowerCase().trim();
  return CATEGORY_IMAGE_MAP[catKey] || "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?auto=format&fit=crop&w=600&q=80";
}

// Initialize Application
document.addEventListener("DOMContentLoaded", async () => {
  setupThemeToggle();
  setupNavigationTabs();
  setupCarouselScroll();
  setupCategoryFilters();
  setupSearchFilter();
  setupModalHandlers();
  setupSimulator();

  await fetchOverview();
  await loadUsers();
  await loadProductsCatalog();
  await refreshDashboard();
});

// Setup Dark/Light Theme Switcher
function setupThemeToggle() {
  const toggleBtn = document.getElementById("btn-theme-toggle");
  const icon = document.getElementById("theme-icon");
  const savedTheme = localStorage.getItem("graphrec_theme") || "dark";
  
  applyTheme(savedTheme);

  toggleBtn.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";
    applyTheme(next);
    localStorage.setItem("graphrec_theme", next);
  });

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    if (theme === "dark") {
      icon.className = "fas fa-moon";
    } else {
      icon.className = "fas fa-sun";
    }
    if (networkInstance) {
      if (currentGraphMode === "ego") loadUserEgoGraph(currentUserId);
      else loadSampleGraph();
    }
  }
}

// Setup Horizontal Navigation Tabs
function setupNavigationTabs() {
  document.querySelectorAll(".nav-tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".view-panel").forEach((v) => (v.style.display = "none"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.style.display = "block";
      }

      if (targetId === "tab-graph") {
        setTimeout(() => {
          if (networkInstance) networkInstance.fit();
        }, 150);
      }
      if (targetId === "tab-recs") {
        renderDedicatedRecsTab();
      }
    });
  });

  // Graph Mode buttons
  document.getElementById("btn-graph-ego").addEventListener("click", () => {
    currentGraphMode = "ego";
    highlightGraphBtn("btn-graph-ego");
    loadUserEgoGraph(currentUserId);
  });

  document.getElementById("btn-graph-sample").addEventListener("click", () => {
    currentGraphMode = "sample";
    highlightGraphBtn("btn-graph-sample");
    loadSampleGraph();
  });
}

function highlightGraphBtn(activeId) {
  ["btn-graph-ego", "btn-graph-sample"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.classList.remove("btn-primary");
  });
  const activeEl = document.getElementById(activeId);
  if (activeEl) activeEl.classList.add("btn-primary");
}

// Horizontal Carousel Smooth Scrolling & Bidirectional Autoscroll
let autoScrollActive = true;
let autoScrollDirection = 1; // 1 = Left-to-Right, -1 = Right-to-Left
let autoScrollSpeed = 1.6;   // Noticeable, smooth cinematic gliding speed
let scrollAccumulator = 0;
let isInteractingWithCard = false;
let autoScrollPauseUntil = 0;
let autoScrollAnimId = null;

function setupCarouselScroll() {
  const track = document.getElementById("product-carousel-track");
  const prevBtn = document.getElementById("scroll-prev-btn");
  const nextBtn = document.getElementById("scroll-next-btn");
  const toggleBtn = document.getElementById("btn-toggle-autoscroll");

  if (!track) return;

  // Initialize accumulator to current scroll
  scrollAccumulator = track.scrollLeft;

  // Sync accumulator on user manual scroll/drag
  track.addEventListener("scroll", () => {
    if (Math.abs(track.scrollLeft - scrollAccumulator) > 4) {
      scrollAccumulator = track.scrollLeft;
    }
  }, { passive: true });

  // Manual scroll buttons (pause autoscroll for 3.5s after click)
  if (prevBtn) {
    prevBtn.addEventListener("click", () => {
      autoScrollPauseUntil = Date.now() + 3500;
      autoScrollDirection = -1;
      track.scrollBy({ left: -320, behavior: "smooth" });
      setTimeout(() => { scrollAccumulator = track.scrollLeft; }, 400);
      updateAutoScrollUI();
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener("click", () => {
      autoScrollPauseUntil = Date.now() + 3500;
      autoScrollDirection = 1;
      track.scrollBy({ left: 320, behavior: "smooth" });
      setTimeout(() => { scrollAccumulator = track.scrollLeft; }, 400);
      updateAutoScrollUI();
    });
  }

  // Toggle button for Play/Pause
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      autoScrollActive = !autoScrollActive;
      if (autoScrollActive) {
        scrollAccumulator = track.scrollLeft;
        autoScrollPauseUntil = 0;
      }
      updateAutoScrollUI();
    });
  }

  // Gentle temporary pause on card hover (2.5 seconds) so user can click, but does NOT get stuck indefinitely
  track.addEventListener("mouseenter", () => {
    autoScrollPauseUntil = Date.now() + 2500;
  });

  // Touch and drag support
  track.addEventListener("pointerdown", (e) => {
    if (e.target.closest("button") || e.target.closest(".product-card")) {
      isInteractingWithCard = true;
    }
  });
  window.addEventListener("pointerup", () => {
    if (isInteractingWithCard) {
      autoScrollPauseUntil = Date.now() + 2000;
      isInteractingWithCard = false;
      scrollAccumulator = track.scrollLeft;
    }
  });

  startAutoScrollEngine();
}

function updateAutoScrollUI() {
  const icon = document.getElementById("autoscroll-icon");
  const text = document.getElementById("autoscroll-text");
  if (!icon || !text) return;

  if (autoScrollActive) {
    icon.className = "fas fa-sync-alt fa-spin";
    icon.style.color = "#10B981";
    text.innerText = autoScrollDirection === 1 ? "Auto-Scroll: L ➔ R" : "Auto-Scroll: R ➔ L";
  } else {
    icon.className = "fas fa-pause";
    icon.style.color = "#94A3B8";
    text.innerText = "Auto-Scroll: Paused";
  }
}

function startAutoScrollEngine() {
  const track = document.getElementById("product-carousel-track");
  if (!track) return;

  let lastTimestamp = performance.now();

  function autoScrollStep(timestamp) {
    const elapsed = Math.min(60, timestamp - lastTimestamp);
    lastTimestamp = timestamp;

    const modal = document.getElementById("product-details-modal");
    const isModalOpen = modal && modal.open;

    const maxScroll = track.scrollWidth - track.clientWidth;

    const canScroll = autoScrollActive && 
                      !isInteractingWithCard && 
                      !isModalOpen && 
                      Date.now() > autoScrollPauseUntil &&
                      maxScroll > 10;

    if (canScroll) {
      // Calculate delta normalized to 60fps
      const delta = autoScrollSpeed * (elapsed / 16.67);

      if (autoScrollDirection === 1) {
        // Scrolling from Left to Right
        scrollAccumulator += delta;
        if (scrollAccumulator >= maxScroll - 3) {
          scrollAccumulator = maxScroll;
          track.scrollLeft = maxScroll;
          autoScrollDirection = -1; // Reverse: now scroll Right to Left!
          autoScrollPauseUntil = Date.now() + 1500; // Pause 1.5s at end
          updateAutoScrollUI();
        } else {
          track.scrollLeft = Math.round(scrollAccumulator);
        }
      } else {
        // Scrolling from Right to Left
        scrollAccumulator -= delta;
        if (scrollAccumulator <= 3) {
          scrollAccumulator = 0;
          track.scrollLeft = 0;
          autoScrollDirection = 1; // Reverse: now scroll Left to Right!
          autoScrollPauseUntil = Date.now() + 1500; // Pause 1.5s at start
          updateAutoScrollUI();
        } else {
          track.scrollLeft = Math.round(scrollAccumulator);
        }
      }
    } else {
      // Keep accumulator in sync if paused or clamped
      if (Math.abs(track.scrollLeft - scrollAccumulator) > 4) {
        scrollAccumulator = track.scrollLeft;
      }
    }

    autoScrollAnimId = requestAnimationFrame(autoScrollStep);
  }

  if (autoScrollAnimId) cancelAnimationFrame(autoScrollAnimId);
  autoScrollAnimId = requestAnimationFrame(autoScrollStep);
}

// Category Filter Pills
function setupCategoryFilters() {
  const container = document.getElementById("category-filter-container");
  container.addEventListener("click", (e) => {
    const btn = e.target.closest(".cat-pill");
    if (!btn) return;

    container.querySelectorAll(".cat-pill").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    currentCategoryFilter = btn.getAttribute("data-category");
    renderProductCarousel();
  });
}

// Search Filter
function setupSearchFilter() {
  const input = document.getElementById("product-search-input");
  input.addEventListener("input", () => {
    renderProductCarousel();
  });
}

// Product Details Modal Handlers
function setupModalHandlers() {
  const modal = document.getElementById("product-details-modal");
  const closeBtn = document.getElementById("modal-close-btn");

  closeBtn.addEventListener("click", () => modal.close());

  // Close when clicking modal backdrop
  modal.addEventListener("click", (e) => {
    const rect = modal.getBoundingClientRect();
    const isInDialog = (
      rect.top <= e.clientY &&
      e.clientY <= rect.top + rect.height &&
      rect.left <= e.clientX &&
      e.clientX <= rect.left + rect.width
    );
    if (!isInDialog) {
      modal.close();
    }
  });
}

// Setup User & Model Controls
function setupSimulator() {
  document.getElementById("user-select").addEventListener("change", async (e) => {
    currentUserId = e.target.value;
    updateUserProfileCard();
    await refreshDashboard();
  });

  document.getElementById("model-select").addEventListener("change", async () => {
    await fetchRecommendations();
  });

  document.getElementById("k-slider").addEventListener("input", (e) => {
    document.getElementById("k-val-display").innerText = e.target.value;
  });

  document.getElementById("k-slider").addEventListener("change", async () => {
    await fetchRecommendations();
  });

  document.getElementById("btn-sim-action").addEventListener("click", async () => {
    const prodId = document.getElementById("sim-product-select").value;
    const action = document.getElementById("sim-action-select").value;
    await triggerInteraction(currentUserId, prodId, action);
  });
}

// Fetch System Overview Metrics
async function fetchOverview() {
  try {
    const res = await fetch("/api/overview");
    const data = await res.json();

    document.getElementById("stat-users").innerText = data.dataset.num_users;
    document.getElementById("stat-products").innerText = data.dataset.num_products;
    document.getElementById("stat-edges").innerText = data.graph.total_edges;
    document.getElementById("stat-interactions").innerText = (data.dataset.train_interactions + data.dataset.test_interactions);
  } catch (err) {
    console.error("Failed to load overview:", err);
  }
}

// Load Users into Selectors
async function loadUsers() {
  try {
    const res = await fetch("/api/users");
    const data = await res.json();
    allUsers = data.users || [];

    const select = document.getElementById("user-select");
    select.innerHTML = "";
    allUsers.forEach((u) => {
      const opt = document.createElement("option");
      opt.value = u.user_id;
      opt.innerText = `${u.user_id} - ${u.name} (${u.primary_interest})`;
      select.appendChild(opt);
    });

    if (allUsers.length > 0) {
      currentUserId = allUsers[0].user_id;
      updateUserProfileCard();
    }
  } catch (err) {
    console.error("Failed to load users:", err);
  }
}

// Load Product Catalog
async function loadProductsCatalog() {
  try {
    const res = await fetch("/api/products");
    const data = await res.json();
    allProducts = data.products || [];

    const simSelect = document.getElementById("sim-product-select");
    simSelect.innerHTML = "";
    allProducts.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.product_id;
      opt.innerText = `[${p.category}] ${p.name}`;
      simSelect.appendChild(opt);
    });

    renderProductCarousel();
  } catch (err) {
    console.error("Failed to load products:", err);
  }
}

// Render Horizontal Scrolling Product Carousel on Homepage
function renderProductCarousel() {
  const track = document.getElementById("product-carousel-track");
  const searchVal = (document.getElementById("product-search-input").value || "").toLowerCase().trim();

  let filtered = allProducts;
  if (currentCategoryFilter !== "ALL") {
    filtered = filtered.filter((p) => p.category === currentCategoryFilter);
  }
  if (searchVal) {
    filtered = filtered.filter((p) => 
      p.name.toLowerCase().includes(searchVal) || 
      p.category.toLowerCase().includes(searchVal) ||
      (p.brand && p.brand.toLowerCase().includes(searchVal))
    );
  }

  document.getElementById("catalog-count-badge").innerText = `${filtered.length} Items`;
  track.innerHTML = "";
  scrollAccumulator = 0;
  track.scrollLeft = 0;
  autoScrollPauseUntil = 0;

  if (filtered.length === 0) {
    track.innerHTML = `<div style="padding:2rem; color:var(--text-muted); font-size:0.9rem;">No products match the selected criteria.</div>`;
    return;
  }

  filtered.forEach((p) => {
    const imgUrl = getProductImageUrl(p);
    const card = document.createElement("div");
    card.className = "product-card";

    // Star rating generator (4.5 to 5.0)
    const rating = (4.5 + ((p.price * 17) % 5) / 10).toFixed(1);
    const reviews = Math.floor(40 + (p.price % 180));

    card.innerHTML = `
      <div class="product-card-img-wrap" onclick="openProductDetailsModal('${p.product_id}')">
        <img class="product-card-img" src="${imgUrl}" alt="${p.name}" loading="lazy" onerror="this.onerror=null; this.src='https://images.unsplash.com/photo-1505740420928-5e560c06d30e?auto=format&fit=crop&w=600&q=80';">
        <span class="product-badge-category">${p.category}</span>
        <span class="product-badge-brand">${p.brand || 'Premium'}</span>
      </div>
      <div class="product-card-body">
        <div class="product-card-title" onclick="openProductDetailsModal('${p.product_id}')">${p.name}</div>
        <div class="product-rating-row">
          <i class="fas fa-star" style="color:#FFFFFF;"></i>
          <span><b>${rating}</b> (${reviews})</span>
        </div>
        <div class="product-card-price">${formatINR(p.price)}</div>
        <div class="product-card-actions">
          <button class="btn-card-action btn-details" onclick="openProductDetailsModal('${p.product_id}')">
            <i class="fas fa-eye"></i> Product Details
          </button>
          <button class="btn-card-action" onclick="triggerInteraction('${currentUserId}', '${p.product_id}', 'cart')">
            <i class="fas fa-shopping-cart"></i> Cart
          </button>
          <button class="btn-card-action" onclick="triggerInteraction('${currentUserId}', '${p.product_id}', 'purchase')">
            <i class="fas fa-bolt"></i> Buy
          </button>
        </div>
      </div>
    `;

    track.appendChild(card);
  });
}

// Open Product Details Modal Dialog
async function openProductDetailsModal(productId) {
  const modal = document.getElementById("product-details-modal");
  const product = allProducts.find((p) => p.product_id === productId);
  if (!product) return;

  const imgUrl = getProductImageUrl(product);
  document.getElementById("modal-img").src = imgUrl;
  document.getElementById("modal-category").innerText = product.category;
  document.getElementById("modal-brand-badge").innerText = `${product.brand || 'Elite Series'} • ${product.product_id}`;
  document.getElementById("modal-title").innerText = product.name;
  document.getElementById("modal-price").innerText = formatINR(product.price);
  document.getElementById("modal-description").innerText = `Premium ${product.category} offering. Featured in multi-hop behavioral embeddings with high collaborative filtering affinity across customer clusters.`;

  // Action buttons
  document.getElementById("modal-btn-cart").onclick = () => {
    triggerInteraction(currentUserId, productId, "cart");
  };
  document.getElementById("modal-btn-buy").onclick = () => {
    triggerInteraction(currentUserId, productId, "purchase");
  };

  // Fetch real-time graph stats for this product
  try {
    const res = await fetch(`/api/product/${productId}`);
    if (res.ok) {
      const data = await res.json();
      const stats = data.graph_stats || {};
      document.getElementById("modal-stat-views").innerText = stats.view || 0;
      document.getElementById("modal-stat-clicks").innerText = stats.search_click || 0;
      document.getElementById("modal-stat-carts").innerText = stats.cart || 0;
      document.getElementById("modal-stat-purchases").innerText = stats.purchase || 0;

      const coContainer = document.getElementById("modal-co-items");
      coContainer.innerHTML = "";
      if (data.co_interacted_items && data.co_interacted_items.length > 0) {
        data.co_interacted_items.forEach((item) => {
          const chip = document.createElement("span");
          chip.className = "chip";
          chip.style.cursor = "pointer";
          chip.innerText = `${item.name} (${item.co_count}x)`;
          chip.onclick = () => openProductDetailsModal(item.product_id);
          coContainer.appendChild(chip);
        });
      } else {
        coContainer.innerHTML = `<span style="font-size:0.75rem; color:var(--text-muted);">None recorded yet</span>`;
      }
    }
  } catch (e) {
    console.error("Failed to load product graph details:", e);
  }

  modal.showModal();
}

// Update Active Customer Profile Card
function updateUserProfileCard() {
  const user = allUsers.find((u) => u.user_id === currentUserId);
  if (!user) return;

  document.getElementById("user-avatar-text").innerText = user.name.charAt(0);
  document.getElementById("user-card-name").innerText = user.name;
  document.getElementById("user-card-id").innerText = `${user.user_id} • Age ${user.age_group} • ${user.activity_level.toUpperCase()}`;

  const chipsContainer = document.getElementById("user-card-interests");
  chipsContainer.innerHTML = "";
  (user.interests || "").split(",").forEach((item) => {
    if (item.trim()) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.innerText = item.trim();
      chipsContainer.appendChild(chip);
    }
  });

  const recSub = document.getElementById("recs-subtitle");
  if (recSub) {
    recSub.innerText = `Personalized for ${user.name} (${user.primary_interest}) based on recursive graph convolutions & PageRank diffusion.`;
  }
}

// Refresh Entire Dashboard State
async function refreshDashboard() {
  await Promise.all([
    fetchUserHistory(currentUserId),
    fetchRecommendations(),
    currentGraphMode === "ego" ? loadUserEgoGraph(currentUserId) : loadSampleGraph(),
    fetchBenchmark()
  ]);
}

// Fetch and Render Horizontal User History
async function fetchUserHistory(userId) {
  try {
    const res = await fetch(`/api/user/${userId}/history`);
    const data = await res.json();
    const container = document.getElementById("user-history-list");
    container.innerHTML = "";

    const countEl = document.getElementById("user-history-count");
    if (countEl) countEl.innerText = `${data.total_interactions || 0} events logged`;

    if (!data.history || data.history.length === 0) {
      container.innerHTML = '<div style="color:var(--text-muted); font-size:0.75rem;">No past interactions logged.</div>';
      return;
    }

    data.history.slice(0, 10).forEach((h) => {
      const chip = document.createElement("div");
      chip.className = "history-chip";

      let badgeClass = "action-view";
      if (h.action_type === "cart") badgeClass = "action-cart";
      if (h.action_type === "purchase") badgeClass = "action-purchase";
      if (h.action_type === "search_click") badgeClass = "action-search";

      chip.innerHTML = `
        <div class="history-chip-header">
          <span style="font-size:0.7rem; color:var(--text-muted);">${h.category}</span>
          <span class="action-badge ${badgeClass}">${h.action_type}</span>
        </div>
        <div style="font-size:0.8rem; font-weight:700; color:var(--text-primary); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
          ${h.name || h.product_id}
        </div>
        <div style="font-size:0.72rem; color:var(--silver-300); margin-top:0.2rem;">
          ${formatINR(h.price)}
        </div>
      `;
      container.appendChild(chip);
    });
  } catch (err) {
    console.error("Failed to load history:", err);
  }
}

// Fetch Personalized Recommendations
let cachedRecommendations = [];

async function fetchRecommendations() {
  const model = document.getElementById("model-select").value;
  const k = document.getElementById("k-slider").value;
  const grid = document.getElementById("recs-container");
  grid.innerHTML = '<div style="color:var(--text-muted); grid-column: 1/-1;">Computing personalized graph embeddings...</div>';

  try {
    const res = await fetch(`/api/recommendations?user_id=${currentUserId}&model_name=${model}&k=${k}`);
    const data = await res.json();
    cachedRecommendations = data.recommendations || [];
    renderRecommendationCards(grid, cachedRecommendations);

    // Also populate dedicated tab container if it exists
    const tabGrid = document.getElementById("tab-recs-container");
    if (tabGrid) {
      renderRecommendationCards(tabGrid, cachedRecommendations);
    }
  } catch (err) {
    console.error("Failed to load recommendations:", err);
    grid.innerHTML = '<div style="color:var(--text-muted); grid-column: 1/-1;">Error computing recommendations.</div>';
  }
}

function renderDedicatedRecsTab() {
  const tabGrid = document.getElementById("tab-recs-container");
  if (tabGrid && cachedRecommendations.length > 0) {
    renderRecommendationCards(tabGrid, cachedRecommendations);
  }
}

function renderRecommendationCards(container, items) {
  container.innerHTML = "";

  if (!items || items.length === 0) {
    container.innerHTML = '<div style="color:var(--text-muted); grid-column: 1/-1;">No recommendations generated.</div>';
    return;
  }

  items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "rec-card";

    const scoreVal = item.adjusted_score !== undefined ? item.adjusted_score : (item.score || 0.5);
    const scorePct = Math.min(100, Math.max(15, Math.round((Math.abs(scoreVal) / (item.score > 1 ? 5 : 1)) * 100)));

    let explanationHtml = "";
    if (item.explanation) {
      explanationHtml = `
        <div class="rec-explanation">
          <i class="fas fa-lightbulb" style="color:#FFFFFF; margin-right:4px;"></i>
          <b>Rationale:</b> ${item.explanation.summary}
        </div>
      `;
    }

    card.innerHTML = `
      <div class="rec-rank-badge">#${item.rank}</div>
      <div class="rec-category">${item.category}</div>
      <div class="rec-title">${item.name}</div>
      <div class="rec-price">${formatINR(item.price)}</div>
      
      <div class="score-bar-wrap">
        <div class="score-bar-labels">
          <span>Graph Affinity Score</span>
          <span>${(scoreVal).toFixed(3)}</span>
        </div>
        <div class="score-bar-bg">
          <div class="score-bar-fill" style="width: ${scorePct}%;"></div>
        </div>
      </div>

      ${explanationHtml}

      <div class="rec-actions">
        <button class="rec-act-btn" onclick="openProductDetailsModal('${item.product_id}')">
          <i class="fas fa-eye"></i> Specs
        </button>
        <button class="rec-act-btn" onclick="triggerInteraction('${currentUserId}', '${item.product_id}', 'cart')">
          <i class="fas fa-shopping-cart"></i> Cart
        </button>
        <button class="rec-act-btn" style="grid-column: 1 / -1;" onclick="inspectExplanationGraph('${currentUserId}', '${item.product_id}')">
          <i class="fas fa-project-diagram"></i> Trace Graph Traversal
        </button>
      </div>
    `;

    container.appendChild(card);
  });
}

// Vis.js Monochrome Black, White & Silver Network Graph
function renderNetwork(graphData) {
  const container = document.getElementById("network-graph-container");
  const isLight = document.documentElement.getAttribute("data-theme") === "light";

  // Palette adaptation for graph
  const userColor = isLight ? "#090A0C" : "#FFFFFF";
  const userFont = isLight ? "#FFFFFF" : "#090A0C";
  const prodColor = isLight ? "#E2E8F0" : "#CBD5E1";
  const prodFont = "#090A0C";
  const catColor = isLight ? "#94A3B8" : "#64748B";

  const nodes = (graphData.nodes || []).map((n) => {
    if (n.group === "user") {
      return { ...n, color: { background: userColor, border: "#CBD5E1" }, font: { color: userFont, face: "Inter", size: 12 } };
    }
    if (n.group === "product") {
      return { ...n, color: { background: prodColor, border: "#94A3B8" }, font: { color: prodFont, face: "Inter", size: 11 } };
    }
    return { ...n, color: { background: catColor, border: "#475569" }, font: { color: "#FFFFFF", face: "Inter", size: 11 } };
  });

  const edges = (graphData.edges || []).map((e) => {
    let edgeColor = isLight ? "#94A3B8" : "#64748B";
    if (e.label === "PURCHASED") edgeColor = isLight ? "#090A0C" : "#FFFFFF";
    if (e.label === "CARTED") edgeColor = isLight ? "#475569" : "#CBD5E1";
    return { ...e, color: { color: edgeColor, highlight: "#FFFFFF" }, font: { color: isLight ? "#475569" : "#94A3B8", size: 9, align: "middle" } };
  });

  const data = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
  const options = {
    nodes: {
      borderWidth: 2,
      shadow: true
    },
    edges: {
      shadow: false,
      smooth: { type: "continuous" }
    },
    physics: {
      stabilization: { iterations: 120 },
      barnesHut: {
        gravitationalConstant: -3200,
        springLength: 90,
        springConstant: 0.04
      }
    },
    interaction: {
      hover: true,
      zoomView: true,
      dragView: true
    }
  };

  if (networkInstance) {
    networkInstance.destroy();
  }

  networkInstance = new vis.Network(container, data, options);

  networkInstance.on("click", (params) => {
    if (params.nodes.length > 0) {
      const clickedId = params.nodes[0];
      if (clickedId.startsWith("P")) {
        openProductDetailsModal(clickedId);
      }
    }
  });
}

async function loadUserEgoGraph(userId) {
  try {
    document.getElementById("graph-title-status").innerText = `Customer Ego-Network (${userId})`;
    const res = await fetch(`/api/graph/user-ego?user_id=${userId}`);
    const data = await res.json();
    renderNetwork(data);
  } catch (err) {
    console.error("Failed to load user ego-graph:", err);
  }
}

async function loadSampleGraph() {
  try {
    document.getElementById("graph-title-status").innerText = "Macro Multi-Relational Catalog Network";
    const res = await fetch("/api/graph/sample");
    const data = await res.json();
    renderNetwork(data);
  } catch (err) {
    console.error("Failed to load sample graph:", err);
  }
}

async function inspectExplanationGraph(userId, productId) {
  try {
    // Switch to graph tab
    document.querySelectorAll(".nav-tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".view-panel").forEach((v) => (v.style.display = "none"));

    const graphTabBtn = document.querySelector('[data-tab="tab-graph"]');
    if (graphTabBtn) graphTabBtn.classList.add("active");
    const graphPanel = document.getElementById("tab-graph");
    if (graphPanel) graphPanel.style.display = "block";

    document.getElementById("graph-title-status").innerText = `Reasoning Path: Customer ${userId} ➔ Product ${productId}`;
    const res = await fetch(`/api/graph/explain?user_id=${userId}&product_id=${productId}`);
    const data = await res.json();
    renderNetwork(data);
  } catch (err) {
    console.error("Failed to load explanation graph:", err);
  }
}

// Trigger Live Customer Behavior Event
async function triggerInteraction(userId, productId, actionType) {
  try {
    const res = await fetch("/api/interact", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: userId,
        product_id: productId,
        action_type: actionType
      })
    });

    const data = await res.json();
    if (res.ok) {
      showToast(`Recorded '${actionType}' on item ${productId} for ${userId}`);
      await refreshDashboard();
    } else {
      showToast(data.detail || "Interaction failed", true);
    }
  } catch (err) {
    console.error("Interaction failed:", err);
    showToast("Network error executing event", true);
  }
}

// Fetch Benchmark Offline Evaluation
async function fetchBenchmark() {
  try {
    const res = await fetch("/api/benchmark");
    const data = await res.json();
    if (!data.results) return;

    const tbody = document.getElementById("benchmark-table-body");
    tbody.innerHTML = "";

    data.results.forEach((row) => {
      const tr = document.createElement("tr");
      const isTop = row["Model"].includes("Hybrid") || row["Model"].includes("LightGCN");
      if (isTop) {
        tr.style.fontWeight = "700";
      }

      tr.innerHTML = `
        <td style="color:var(--text-primary);"><i class="fas ${isTop ? 'fa-award' : 'fa-circle-notch'}" style="margin-right:6px; font-size:0.75rem;"></i>${row["Model"]}</td>
        <td>${Number(row["NDCG@10"]).toFixed(4)}</td>
        <td>${Number(row["HitRate@10"]).toFixed(4)}</td>
        <td>${Number(row["Precision@10"]).toFixed(4)}</td>
        <td>${Number(row["Recall@10"]).toFixed(4)}</td>
        <td>${(Number(row["CatalogCoverage"]) * 100).toFixed(1)}%</td>
        <td>${Number(row["Diversity"]).toFixed(3)}</td>
      `;
      tbody.appendChild(tr);
    });

    if (data.chart_url) {
      document.getElementById("benchmark-chart-img").src = `${data.chart_url}?t=${Date.now()}`;
    }
  } catch (err) {
    console.error("Failed to load benchmark:", err);
  }
}

// Floating Toast Notification
let toastTimeout = null;
function showToast(msg, isError = false) {
  const toast = document.getElementById("toast-notification");
  const text = document.getElementById("toast-text");
  text.innerText = msg;
  toast.classList.add("show");

  if (toastTimeout) clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    toast.classList.remove("show");
  }, 3500);
}
