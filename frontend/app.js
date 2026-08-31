/* ============================================================
   走遍中国 · 智能旅游助手 — 交互逻辑
   API 契约保持不变：/api/cities /api/ask /api/health
   ============================================================ */

// API 地址自动适配：手机访问时用同源地址，本地文件打开时用 localhost
const API_BASE = window.location.origin.startsWith('http')
    ? window.location.origin
    : 'http://localhost:8000';

// 状态
const compareMode = location.pathname.includes('/compare');
const HISTORY_KEY = 'travel_qa_history';
const MAX_HISTORY = 20;

const CITY_EMOJI_MAP = {
    '北京': '🏯', '杭州': '🌸', '成都': '🐼', '西安': '⚔️',
    '重庆': '🌆', '广州': '🥟', '苏州': '🏡', '长沙': '🌶️',
};
const METHOD_LABELS = { bm25: 'BM25', keyword: '关键词', vector: '向量', hybrid: '混合' };

// DOM
const $ = (id) => document.getElementById(id);
const els = {
    chatScroll: $('chat-scroll'),
    conversation: $('conversation'),
    welcome: $('welcome'),
    welcomeChips: $('welcome-chips'),
    cityChips: $('city-chips'),
    methodToggle: $('method-toggle'),
    compareBanner: $('compare-banner'),
    questionInput: $('question-input'),
    askBtn: $('ask-btn'),
    clearBtn: $('clear-btn'),
    historyBtn: $('history-btn'),
    historyClear: $('history-clear'),
    typing: $('typing'),
    typingText: $('typing-text'),
    quickTags: $('quick-tags'),
    compareCard: $('compare-card'),
    toast: $('toast'),
    statusPill: $('status-pill'),
    statusLabel: $('status-label'),
    sheet: $('history-sheet'),
    sheetBackdrop: $('sheet-backdrop'),
    historyList: $('history-list'),
    contributeToggle: $('contribute-toggle'),
    contributePanel: $('contribute-panel'),
    contributeCity: $('contribute-city'),
    contributeTitle: $('contribute-title'),
    contributeContent: $('contribute-content'),
    contributeFile: $('contribute-file'),
    submitContribution: $('submit-contribution'),
    contributeStatus: $('contribute-status'),
};

let currentMethod = 'bm25';
let currentCity = '';
let abortController = null;
let typingTimer = null;
let toastTimer = null;
let healthNotified = false;
window._cityList = []; // 全局城市名缓存，供 extractCityFromQuestion 使用

// 对比模式：仅通过 /compare 地址进入
if (compareMode) {
    els.compareBanner.hidden = false;
    els.methodToggle.classList.add('dimmed');
}

// ========== 初始化 ==========
async function init() {
    await loadCities();   // 先从后端加载城市列表
    loadHistory();
    bindEvents();
    checkHealth();
}

// ========== 从后端动态加载城市信息 ==========
async function loadCities() {
    try {
        const res = await fetch(`${API_BASE}/api/cities`);
        if (!res.ok) throw new Error('Failed');
        const cities = await res.json();

        // 缓存城市名列表
        window._cityList = cities.map(c => c.city);

        // 1. 渲染城市 chips（全部 + 各城市）
        els.cityChips.innerHTML = '';
        els.cityChips.appendChild(makeChip('🌍', '全部', ''));
        cities.forEach(c => {
            const emoji = CITY_EMOJI_MAP[c.city] || '📍';
            els.cityChips.appendChild(makeChip(emoji, c.city, c.city));
        });

        // 2. 动态生成快捷标签（从各城市采样热门关键词）
        const sampleQueries = [
            { q: '北京故宫门票多少钱？', label: '北京故宫' },
            { q: '成都火锅有什么推荐？', label: '成都火锅' },
            { q: '西安兵马俑怎么去？', label: '西安兵马俑' },
            { q: '长沙橘子洲怎么玩？', label: '长沙橘子洲' },
            { q: '广州早茶推荐什么？', label: '广州早茶' },
            { q: '苏州园林一天能逛几个？', label: '苏州园林' },
            { q: '重庆三天两晚怎么安排？', label: '重庆三日游' },
            { q: '杭州西湖有哪些景点？', label: '杭州西湖' },
        ];
        // 只展示已覆盖城市的快捷标签
        const coveredCities = cities.map(c => c.city);
        const tagsContainer = els.quickTags;
        tagsContainer.innerHTML = '';
        sampleQueries.forEach(item => {
            const matchedCity = coveredCities.find(city => item.q.includes(city));
            if (matchedCity || item.q.includes('三天') || item.q.includes('早茶')) {
                const tag = document.createElement('button');
                tag.type = 'button';
                tag.className = 'chip quick-tag';
                tag.dataset.q = item.q;
                tag.textContent = item.label;
                tagsContainer.appendChild(tag);
            }
        });

        // 3. 欢迎页示例问题（取前 4 个快捷标签）
        els.welcomeChips.innerHTML = '';
        [...tagsContainer.children].slice(0, 4).forEach(tag => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'welcome-chip';
            btn.dataset.q = tag.dataset.q;
            btn.textContent = tag.textContent;
            els.welcomeChips.appendChild(btn);
        });

        console.log(`✅ ${cities.length} 个城市已加载：${window._cityList.join('、')}`);
    } catch (e) {
        console.warn('⚠️ 无法加载城市列表，使用默认值', e);
    }
}

function makeChip(emoji, label, value) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip';
    chip.dataset.city = value || '';
    chip.innerHTML = `<span class="chip-emoji">${emoji}</span><span>${label}</span>`;
    return chip;
}

// ========== 事件绑定 ==========
function bindEvents() {
    // 提问
    els.askBtn.addEventListener('click', () => askQuestion());

    // 清空对话
    els.clearBtn.addEventListener('click', clearAll);

    // 历史抽屉
    els.historyBtn.addEventListener('click', toggleSheet);
    els.sheetBackdrop.addEventListener('click', closeSheet);
    els.historyClear.addEventListener('click', () => {
        try { localStorage.removeItem(HISTORY_KEY); } catch (e) { /* ignore */ }
        renderHistory();
        showToast('问答记录已清空', 'ok');
    });

    // 检索方式切换
    els.methodToggle.addEventListener('click', (e) => {
        const btn = e.target.closest('.method-option');
        if (!btn) return;
        currentMethod = btn.dataset.method;
        els.methodToggle.querySelectorAll('.method-option').forEach(b => b.classList.toggle('active', b === btn));
    });

    // 回车键提问
    els.questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            askQuestion();
        }
    });

    // 城市 chips 点击
    els.cityChips.addEventListener('click', (e) => {
        const chip = e.target.closest('.chip');
        if (!chip) return;
        const city = chip.dataset.city || '';
        setCityFilter(city);
        if (city && !els.questionInput.value.trim()) {
            els.questionInput.value = `${city}有什么好玩的地方？`;
        }
        els.questionInput.focus();
    });

    // 快捷标签点击
    els.quickTags.addEventListener('click', (e) => {
        const tag = e.target.closest('.quick-tag');
        if (!tag) return;
        runQuickQuery(tag.dataset.q);
    });

    // 欢迎页示例点击
    els.welcomeChips.addEventListener('click', (e) => {
        const btn = e.target.closest('.welcome-chip');
        if (!btn) return;
        runQuickQuery(btn.dataset.q);
    });

    // 知识贡献开关
    els.contributeToggle.addEventListener('click', () => {
        const hidden = els.contributePanel.hidden;
        els.contributePanel.hidden = !hidden;
        if (!els.contributePanel.hidden) {
            els.contributeCity.focus();
        }
    });

    els.submitContribution.addEventListener('click', submitContribution);
}

function runQuickQuery(q) {
    els.questionInput.value = q;
    const city = extractCityFromQuestion(q);
    if (city) setCityFilter(city);
    askQuestion();
}

function setCityFilter(city) {
    currentCity = city;
    els.cityChips.querySelectorAll('.chip').forEach(c => {
        c.classList.toggle('active', c.dataset.city === city);
    });
}

// ========== 辅助函数 ==========
function extractCityFromQuestion(q) {
    const cities = window._cityList || [];
    for (const city of cities) {
        if (q.includes(city)) return city;
    }
    return '';
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        els.chatScroll.scrollTo({ top: els.chatScroll.scrollHeight, behavior: 'smooth' });
    });
}

// ========== 打字指示器 ==========
function showTyping(mode) {
    els.typing.hidden = false;
    const stages = mode === 'compare'
        ? ['正在并行检索四种方式…', '正在汇总召回结果…']
        : ['正在检索知识库…', '正在组织答案…', '马上就好…'];
    let i = 0;
    els.typingText.textContent = stages[0];
    clearInterval(typingTimer);
    typingTimer = setInterval(() => {
        i = (i + 1) % stages.length;
        els.typingText.textContent = stages[i];
    }, 2400);
    scrollToBottom();
}

function hideTyping() {
    els.typing.hidden = true;
    clearInterval(typingTimer);
}

// ========== Toast ==========
function showToast(text, type = 'info', retryable = false) {
    const icons = { info: 'ℹ️', warn: '⚠️', error: '⚠️', ok: '✅' };
    let html = `<span class="toast-icon">${icons[type] || 'ℹ️'}</span><span class="toast-text">${text}</span>`;
    if (retryable) html += `<button type="button" class="toast-btn" id="toast-retry">重试</button>`;
    els.toast.className = 'toast ' + type;
    els.toast.innerHTML = html;
    els.toast.hidden = false;
    const retry = els.toast.querySelector('#toast-retry');
    if (retry) retry.addEventListener('click', () => { hideToast(); askQuestion(); });
    clearTimeout(toastTimer);
    if (type !== 'error') toastTimer = setTimeout(hideToast, 4200);
}

function hideToast() {
    els.toast.hidden = true;
    clearTimeout(toastTimer);
}

// ========== 轻量 Markdown 渲染（先转义防 XSS，再解析格式） ==========
function renderMarkdown(text) {
    if (!text) return '';
    const lines = escapeHtml(text).split('\n');
    let html = '';
    let listType = null; // 'ul' | 'ol'
    let inCode = false;
    let codeBuf = [];

    const closeList = () => {
        if (listType) { html += `</${listType}>`; listType = null; }
    };

    for (const rawLine of lines) {
        // 代码块
        if (/^\s*```/.test(rawLine)) {
            if (inCode) {
                html += '<pre class="md-code">' + codeBuf.join('\n') + '</pre>';
                codeBuf = [];
                inCode = false;
            } else {
                closeList();
                inCode = true;
            }
            continue;
        }
        if (inCode) { codeBuf.push(rawLine); continue; }

        // 标题
        const h = rawLine.match(/^(#{1,4})\s+(.*)/);
        if (h) { closeList(); html += `<h${h[1].length}>${inlineMd(h[2])}</h${h[1].length}>`; continue; }

        // 无序列表
        const ul = rawLine.match(/^\s*[-*+]\s+(.*)/);
        if (ul) {
            if (listType !== 'ul') { closeList(); listType = 'ul'; html += '<ul>'; }
            html += `<li>${inlineMd(ul[1])}</li>`;
            continue;
        }

        // 有序列表
        const ol = rawLine.match(/^\s*\d+[.、)]\s+(.*)/);
        if (ol) {
            if (listType !== 'ol') { closeList(); listType = 'ol'; html += '<ol>'; }
            html += `<li>${inlineMd(ol[1])}</li>`;
            continue;
        }

        // 引用
        const bq = rawLine.match(/^\s*>\s?(.*)/);
        if (bq) { closeList(); html += `<blockquote>${inlineMd(bq[1])}</blockquote>`; continue; }

        // 分隔线
        if (/^\s*([-*_]\s*){3,}\s*$/.test(rawLine)) { closeList(); html += '<hr>'; continue; }

        closeList();
        if (rawLine.trim() === '') { continue; }
        html += `<p>${inlineMd(rawLine)}</p>`;
    }

    if (inCode) html += '<pre class="md-code">' + codeBuf.join('\n') + '</pre>';
    closeList();
    return html;
}

function inlineMd(s) {
    return s
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/__([^_]+)__/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
}

// ========== 消息渲染 ==========
function pushUserMessage(text) {
    els.welcome.hidden = true;
    const wrap = document.createElement('div');
    wrap.className = 'msg msg-user';
    const bubble = document.createElement('div');
    bubble.className = 'bubble bubble-user';
    bubble.textContent = text;
    wrap.appendChild(bubble);
    els.conversation.appendChild(wrap);
    scrollToBottom();
}

function pushAiMessage(data) {
    els.welcome.hidden = true;
    const wrap = document.createElement('div');
    wrap.className = 'msg msg-ai';

    let meta = '';
    if (data.detected_city) meta += `<span>📍 ${escapeHtml(data.detected_city)}</span>`;
    if (data.retrieval_method) meta += `<span>🔍 ${escapeHtml(METHOD_LABELS[data.retrieval_method] || data.retrieval_method)}</span>`;
    if (data.model) meta += `<span>🤖 ${escapeHtml(data.model)}</span>`;

    let sources = '';
    if (data.sources && data.sources.length > 0) {
        sources = '<div class="bubble-sources">';
        data.sources.forEach((src, idx) => {
            sources += `<span class="src-chip"><i>${idx + 1}</i><span class="src-title">${escapeHtml(src.title || '')}</span>${src.city ? `<em>${escapeHtml(src.city)}</em>` : ''}</span>`;
        });
        sources += '</div>';
    }

    wrap.innerHTML = `
            <div class="ai-avatar">
                <svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 2.5l2.3 7.2 7.2 2.3-7.2 2.3L12 21.5l-2.3-7.2-7.2-2.3 7.2-2.3z"/>
                    <path d="M19 3.5l.9 2.6 2.6.9-2.6.9-.9 2.6-.9-2.6-2.6-.9 2.6-.9z"/>
                </svg>
            </div>
            <div class="bubble bubble-ai">
                <div class="bubble-head">
                    <span class="ai-name">走遍中国助手</span>
                    <span class="ai-time">${formatTime(new Date().toISOString())}</span>
                </div>
                <div class="bubble-body"></div>
                <div class="bubble-meta">${meta}</div>
                ${sources}
            </div>`;
    wrap.querySelector('.bubble-body').innerHTML = renderMarkdown(data.answer || '');
    els.conversation.appendChild(wrap);
    scrollToBottom();
}

// ========== 对比矩阵渲染 ==========
function renderCompare(data) {
    const METHODS = [
        { method: 'keyword', label: '关键词' },
        { method: 'bm25', label: 'BM25' },
        { method: 'vector', label: '向量' },
        { method: 'hybrid', label: '混合' },
    ];

    // 按来源 id 聚合四种方式的召回
    const byId = new Map();
    data.methods.forEach(m => {
        if (m.status !== 'ok') return;
        m.results.forEach((r, idx) => {
            const id = r.id || r.title || '';
            if (!byId.has(id)) {
                byId.set(id, { id: id, title: r.title || '', city: r.city || '', foundBy: [] });
            }
            byId.get(id).foundBy.push({ method: m.method, rank: idx + 1 });
        });
    });
    const sources = Array.from(byId.values());
    sources.forEach(s => {
        s.bestRank = Math.min(...s.foundBy.map(f => f.rank));
        s.foundCount = s.foundBy.length;
    });
    // 被越多方式召回越靠前，其次看最佳名次
    sources.sort((a, b) => (b.foundCount - a.foundCount) || (a.bestRank - b.bestRank));

    // 摘要：Top1 一致性
    const okMethods = data.methods.filter(m => m.status === 'ok' && m.results.length > 0);
    const top1Counts = {};
    okMethods.forEach(m => {
        const id = m.results[0].id || m.results[0].title || '';
        top1Counts[id] = (top1Counts[id] || 0) + 1;
    });
    const top1Best = Math.max(0, ...Object.values(top1Counts));

    let html = '<div class="compare-head"><span>⚖️</span> 检索方式对比';
    if (data.detected_city) {
        html += ` <span class="compare-city">${escapeHtml(data.detected_city)}</span>`;
    }
    html += '</div>';

    html += '<div class="compare-summary">';
    html += `<span class="cs-item">召回来源 <b>${sources.length}</b> 条</span>`;
    if (okMethods.length > 0) {
        html += `<span class="cs-item">Top1 一致 <b>${top1Best}/${okMethods.length}</b></span>`;
    }
    html += '</div>';

    html += '<div class="compare-chips">';
    data.methods.forEach(m => {
        const text = m.status === 'ok'
            ? `${m.label} ${m.latency_ms}ms`
            : `${m.label} ${m.status === 'unavailable' ? '不可用' : '出错'}`;
        html += `<span class="cc-chip ${m.status === 'ok' ? 'cc-' + m.method : 'cc-bad'}">${escapeHtml(text)}</span>`;
    });
    html += '</div>';

    html += '<div class="cm-table">';
    html += '<div class="cm-row cm-head"><div class="cm-source">来源</div>';
    METHODS.forEach(m => { html += `<div class="cm-col">${m.label}</div>`; });
    html += '</div>';

    if (sources.length === 0) {
        html += '<div class="cm-empty">无召回结果</div>';
    } else {
        sources.forEach((s, idx) => {
            html += `<div class="cm-row${s.foundCount > 1 ? ' cm-shared' : ''}">`;
            html += `<div class="cm-source"><span class="cm-src-rank">${idx + 1}</span><span class="cm-title" title="${escapeHtml(s.title)}">${escapeHtml(s.title)}</span>${s.city ? `<span class="cm-city">${escapeHtml(s.city)}</span>` : ''}</div>`;
            METHODS.forEach(m => {
                const f = s.foundBy.find(x => x.method === m.method);
                if (!f) {
                    html += '<div class="cm-col"><span class="cm-miss">—</span></div>';
                } else {
                    html += `<div class="cm-col"><span class="cm-rank r-${m.method}">${f.rank}</span></div>`;
                }
            });
            html += '</div>';
        });
    }
    html += '</div>';

    html += '<div class="compare-legend">数字＝该方式中的名次 · —＝未召回 · <span class="lg-shared">高亮行</span>＝多种方式共同召回</div>';

    els.compareCard.innerHTML = html;
    els.compareCard.hidden = false;
    scrollToBottom();
}

// ========== 对比模式：分别请求四种检索方式 ==========
async function runCompare(question, city) {
    const methods = [
        { method: 'keyword', label: '关键词匹配' },
        { method: 'bm25', label: 'BM25' },
        { method: 'vector', label: '向量相似度' },
        { method: 'hybrid', label: 'BM25+向量' },
    ];

    let vectorStatus = 'not_loaded';
    try {
        const h = await fetch(`${API_BASE}/api/health`);
        if (h.ok) {
            vectorStatus = (await h.json()).vector_retrieval || 'not_loaded';
        }
    } catch (e) { /* 忽略 */ }

    const out = [];
    let detectedCity = city || null;
    for (const m of methods) {
        const t0 = performance.now();
        let status = 'ok';
        let results = [];
        try {
            const res = await fetch(`${API_BASE}/api/ask`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: question,
                    city: city,
                    top_k: 3,
                    method: m.method,
                    raw: true,
                }),
                signal: abortController.signal,
            });
            if (!res.ok) throw new Error('http ' + res.status);
            const data = await res.json();
            if (!detectedCity && data.detected_city) {
                detectedCity = data.detected_city;
            }
            results = data.sources || [];
        } catch (e) {
            if (e.name === 'AbortError') throw e;
            status = 'error';
        }
        if ((m.method === 'vector' || m.method === 'hybrid') && vectorStatus === 'unavailable') {
            status = 'unavailable';
            results = [];
        }
        out.push({
            method: m.method,
            label: m.label,
            status: status,
            latency_ms: Math.round(performance.now() - t0),
            results: results,
        });
    }
    return { question: question, detected_city: detectedCity, methods: out };
}

// ========== 核心：提问 ==========
async function askQuestion() {
    const question = els.questionInput.value.trim();

    // 空问题检查
    if (!question) {
        showToast('请输入您的问题', 'warn');
        els.questionInput.focus();
        return;
    }

    // 敏感词简单过滤
    const sensitivePatterns = /[黄赌毒]|赌博|色情|违法/;
    if (sensitivePatterns.test(question)) {
        showToast('请提出与旅游相关的问题', 'warn');
        return;
    }

    // UI 状态
    hideToast();
    els.questionInput.value = '';
    els.compareCard.hidden = true;
    showTyping(compareMode ? 'compare' : 'normal');
    els.askBtn.disabled = true;

    // 取消之前的请求
    if (abortController) {
        abortController.abort();
    }
    abortController = new AbortController();

    const city = currentCity || null;

    try {
        let data;
        if (compareMode) {
            data = await runCompare(question, city);
        } else {
            const res = await fetch(`${API_BASE}/api/ask`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: question,
                    city: city,
                    top_k: 5,
                    method: currentMethod
                }),
                signal: abortController.signal,
            });

            if (!res.ok) {
                if (res.status === 400) {
                    const errData = await res.json();
                    throw new Error(errData.detail || '输入有误');
                } else if (res.status === 422) {
                    const errData = await res.json();
                    const msg = errData.detail?.[0]?.msg || '输入格式有误';
                    throw new Error(msg);
                } else if (res.status >= 500) {
                    throw new Error('服务器内部错误，请稍后重试');
                }
                throw new Error(`请求失败 (${res.status})`);
            }
            data = await res.json();
        }

        // 先上屏用户问题
        pushUserMessage(question);

        if (compareMode) {
            renderCompare(data);
        } else {
            pushAiMessage(data);
            saveToHistory(question, data);
            renderHistory();
        }

    } catch (err) {
        if (err.name === 'AbortError') {
            // 用户取消，不报错
            return;
        }
        if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
            showError('请求失败，请确认后端已启动（localhost:8000）<br><small>启动命令：<code>cd backend &amp;&amp; uvicorn main:app --host 0.0.0.0 --port 8000</code></small>');
        } else if (err.message.includes('timeout') || err.message.includes('Timeout')) {
            showError('服务器响应超时，请重试');
        } else {
            showError(escapeHtml(err.message));
        }
    } finally {
        hideTyping();
        els.askBtn.disabled = false;
        abortController = null;
    }
}

async function submitContribution() {
    const city = els.contributeCity.value.trim();
    const title = els.contributeTitle.value.trim();
    const content = els.contributeContent.value.trim();
    const file = els.contributeFile.files && els.contributeFile.files[0];

    if (!city) {
        setContributionStatus('请输入城市名称', 'warn');
        return;
    }
    if (!content && !file) {
        setContributionStatus('请填写文案或上传文件', 'warn');
        return;
    }

    const formData = new FormData();
    formData.append('city', city);
    formData.append('title', title || `${city}旅游体验`);
    formData.append('content', content);
    formData.append('source', '用户亲身经历');
    formData.append('source_type', file ? 'file' : 'text');
    formData.append('notes', `来自前端上传：${title || city}`);
    if (file) formData.append('file', file);

    setContributionStatus('正在审核并整理知识…', 'info');
    els.submitContribution.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/api/contribute`, {
            method: 'POST',
            body: formData,
        });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.detail || '提交失败');
        }

        if (data.status === 'approved') {
            setContributionStatus(`${data.reason || '提交成功'}，已纳入知识库。`, 'ok');
            els.contributeContent.value = '';
            els.contributeTitle.value = '';
            els.contributeCity.value = '';
            els.contributeFile.value = '';
            els.contributePanel.hidden = true;
        } else {
            setContributionStatus(data.reason || '提交未通过审核', 'warn');
        }
    } catch (err) {
        setContributionStatus(err.message || '提交失败，请稍后再试', 'error');
    } finally {
        els.submitContribution.disabled = false;
    }
}

function setContributionStatus(message, type = 'info') {
    els.contributeStatus.textContent = message;
    els.contributeStatus.className = `contribute-status ${type}`;
}

function showError(text) {
    showToast(text, 'error', true);
}

// ========== 清空对话 ==========
function clearAll() {
    if (abortController) abortController.abort();
    els.conversation.innerHTML = '';
    els.compareCard.hidden = true;
    els.questionInput.value = '';
    setCityFilter('');
    els.welcome.hidden = false;
    hideTyping();
    hideToast();
    els.chatScroll.scrollTop = 0;
}

// ========== 后端健康检查 ==========
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(3000) });
        if (res.ok) {
            setStatus('online');
            console.log('✅ 后端连接成功');
        } else {
            setStatus('offline');
        }
    } catch (e) {
        setStatus('offline');
        console.warn('⚠️ 后端未启动，请执行: cd backend && uvicorn main:app --host 0.0.0.0 --port 8000');
        if (!healthNotified) {
            healthNotified = true;
            showToast('后端服务未启动<br><small>请执行：<code>cd backend &amp;&amp; uvicorn main:app --host 0.0.0.0 --port 8000</code></small>', 'warn');
        }
    }
}

function setStatus(s) {
    els.statusPill.classList.remove('online', 'offline');
    if (s === 'online') {
        els.statusPill.classList.add('online');
        els.statusLabel.textContent = '在线';
    } else {
        els.statusPill.classList.add('offline');
        els.statusLabel.textContent = '离线';
    }
}

// ========== 问答历史 (localStorage) ==========
function getHistory() {
    try {
        const raw = localStorage.getItem(HISTORY_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch (e) {
        return [];
    }
}

function saveToHistory(question, data) {
    const history = getHistory();
    history.unshift({
        question: question,
        answer: data.answer ? data.answer.substring(0, 100) + '...' : '',
        detected_city: data.detected_city,
        timestamp: new Date().toISOString(),
        fullData: data,
    });
    // 限制数量
    if (history.length > MAX_HISTORY) {
        history.length = MAX_HISTORY;
    }
    try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch (e) {
        // localStorage 满，清除旧记录
        history.length = 10;
        localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    }
}

function renderHistory() {
    const history = getHistory();
    if (history.length === 0) {
        els.historyList.innerHTML = '<div class="h-empty">暂无问答记录</div>';
        return;
    }
    els.historyList.innerHTML = history.slice(0, 10).map((item, idx) => `
            <div class="history-item" data-idx="${idx}">
                <div class="h-item-q">${escapeHtml(item.question)}</div>
                <div class="h-item-meta">${item.detected_city ? '📍 ' + escapeHtml(item.detected_city) + ' · ' : ''}${formatTime(item.timestamp)}</div>
            </div>
        `).join('');

    // 点击历史记录恢复
    els.historyList.querySelectorAll('.history-item').forEach(el => {
        el.addEventListener('click', () => {
            const idx = parseInt(el.dataset.idx);
            const history = getHistory();
            const item = history[idx];
            if (item && item.fullData) {
                els.questionInput.value = item.question;
                if (item.fullData.detected_city) {
                    setCityFilter(item.fullData.detected_city);
                }
                closeSheet();
                pushUserMessage(item.question);
                pushAiMessage(item.fullData);
            }
        });
    });
}

function loadHistory() {
    renderHistory();
}

// ========== 历史抽屉开关 ==========
function toggleSheet() {
    const open = !els.sheet.classList.contains('open');
    els.sheet.classList.toggle('open', open);
    els.sheetBackdrop.classList.toggle('show', open);
    if (open) renderHistory();
}

function closeSheet() {
    els.sheet.classList.remove('open');
    els.sheetBackdrop.classList.remove('show');
}

// ========== 工具函数 ==========
function formatTime(isoStr) {
    try {
        const d = new Date(isoStr);
        const now = new Date();
        const diff = now - d;
        if (diff < 60000) return '刚刚';
        if (diff < 3600000) return Math.floor(diff / 60000) + '分钟前';
        if (diff < 86400000) return Math.floor(diff / 3600000) + '小时前';
        return d.toLocaleDateString('zh-CN');
    } catch (e) {
        return '';
    }
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ========== 启动 ==========
init();
console.log('🏔️ 走遍中国 · 智能旅游助手 已就绪');
console.log('💡 快捷操作：点击城市卡片、快捷标签，或直接输入问题后按回车');
