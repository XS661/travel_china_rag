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

// 城市名 → 省级行政区名，用于把用户投稿定位到中国地图上的省份
const CITY_PROVINCE_MAP = {
    '北京': '北京市',
    '上海市': '上海市', '重庆': '重庆市', '天津': '天津市',
    '河北': '河北省', '石家庄': '河北省', '唐山': '河北省', '秦皇岛': '河北省', '邯郸': '河北省', '邢台': '河北省', '保定': '河北省', '张家口': '河北省', '承德': '河北省', '沧州': '河北省', '廊坊': '河北省', '衡水': '河北省',
    '山西': '山西省', '太原': '山西省', '大同': '山西省', '阳泉': '山西省', '长治': '山西省', '晋城': '山西省', '朔州': '山西省', '晋中': '山西省', '运城': '山西省', '忻州': '山西省', '临汾': '山西省', '吕梁': '山西省',
    '内蒙古': '内蒙古自治区', '呼和浩特': '内蒙古自治区', '呼伦贝尔': '内蒙古自治区', '包头': '内蒙古自治区', '乌海': '内蒙古自治区', '赤峰': '内蒙古自治区', '通辽': '内蒙古自治区', '鄂尔多斯': '内蒙古自治区', '巴彦淖尔': '内蒙古自治区', '乌兰察布': '内蒙古自治区',
    '辽宁': '辽宁省', '沈阳': '辽宁省', '大连': '辽宁省', '鞍山': '辽宁省', '抚顺': '辽宁省', '本溪': '辽宁省', '丹东': '辽宁省', '锦州': '辽宁省', '营口': '辽宁省', '阜新': '辽宁省', '辽阳': '辽宁省', '盘锦': '辽宁省', '铁岭': '辽宁省', '朝阳': '辽宁省', '葫芦岛': '辽宁省',
    '吉林': '吉林省', '长春': '吉林省', '吉林市': '吉林省', '四平': '吉林省', '辽源': '吉林省', '通化': '吉林省', '白山': '吉林省', '松原': '吉林省', '白城': '吉林省', '延边': '吉林省',
    '黑龙江': '黑龙江省', '哈尔滨': '黑龙江省', '齐齐哈尔': '黑龙江省', '鸡西': '黑龙江省', '鹤岗': '黑龙江省', '双鸭山': '黑龙江省', '大庆': '黑龙江省', '伊春': '黑龙江省', '佳木斯': '黑龙江省', '七台河': '黑龙江省', '牡丹江': '黑龙江省', '黑河': '黑龙江省', '绥化': '黑龙江省',
    '江苏': '江苏省', '南京': '江苏省', '无锡': '江苏省', '徐州': '江苏省', '常州': '江苏省', '苏州': '江苏省', '南通': '江苏省', '连云港': '江苏省', '淮安': '江苏省', '盐城': '江苏省', '扬州': '江苏省', '镇江': '江苏省', '泰州': '江苏省', '宿迁': '江苏省',
    '浙江': '浙江省', '杭州': '浙江省', '宁波': '浙江省', '温州': '浙江省', '嘉兴': '浙江省', '湖州': '浙江省', '绍兴': '浙江省', '金华': '浙江省', '衢州': '浙江省', '舟山': '浙江省', '台州': '浙江省', '丽水': '浙江省',
    '安徽': '安徽省', '合肥': '安徽省', '芜湖': '安徽省', '蚌埠': '安徽省', '淮南': '安徽省', '马鞍山': '安徽省', '淮北': '安徽省', '铜陵': '安徽省', '安庆': '安徽省', '黄山': '安徽省', '滁州': '安徽省', '阜阳': '安徽省', '宿州': '安徽省', '六安': '安徽省', '亳州': '安徽省', '池州': '安徽省', '宣城': '安徽省',
    '福建': '福建省', '福州': '福建省', '厦门': '福建省', '莆田': '福建省', '三明': '福建省', '泉州': '福建省', '漳州': '福建省', '南平': '福建省', '龙岩': '福建省', '宁德': '福建省',
    '江西': '江西省', '南昌': '江西省', '景德镇': '江西省', '萍乡': '江西省', '九江': '江西省', '新余': '江西省', '鹰潭': '江西省', '赣州': '江西省', '吉安': '江西省', '宜春': '江西省', '抚州': '江西省', '上饶': '江西省',
    '山东': '山东省', '济南': '山东省', '青岛': '山东省', '淄博': '山东省', '枣庄': '山东省', '东营': '山东省', '烟台': '山东省', '潍坊': '山东省', '济宁': '山东省', '泰安': '山东省', '威海': '山东省', '日照': '山东省', '临沂': '山东省', '德州': '山东省', '聊城': '山东省', '滨州': '山东省', '菏泽': '山东省',
    '河南': '河南省', '郑州': '河南省', '开封': '河南省', '洛阳': '河南省', '平顶山': '河南省', '安阳': '河南省', '鹤壁': '河南省', '新乡': '河南省', '焦作': '河南省', '濮阳': '河南省', '许昌': '河南省', '漯河': '河南省', '三门峡': '河南省', '南阳': '河南省', '商丘': '河南省', '信阳': '河南省', '周口': '河南省', '驻马店': '河南省',
    '湖北': '湖北省', '武汉': '湖北省', '黄石': '湖北省', '十堰': '湖北省', '宜昌': '湖北省', '襄阳': '湖北省', '鄂州': '湖北省', '荆门': '湖北省', '孝感': '湖北省', '荆州': '湖北省', '黄冈': '湖北省', '咸宁': '湖北省', '随州': '湖北省',
    '湖南': '湖南省', '长沙': '湖南省', '株洲': '湖南省', '湘潭': '湖南省', '衡阳': '湖南省', '邵阳': '湖南省', '岳阳': '湖南省', '常德': '湖南省', '张家界': '湖南省', '益阳': '湖南省', '郴州': '湖南省', '永州': '湖南省', '怀化': '湖南省', '娄底': '湖南省', '湘西': '湖南省',
    '广东': '广东省', '广州': '广东省', '韶关': '广东省', '深圳': '广东省', '珠海': '广东省', '汕头': '广东省', '佛山': '广东省', '江门': '广东省', '湛江': '广东省', '茂名': '广东省', '肇庆': '广东省', '惠州': '广东省', '梅州': '广东省', '汕尾': '广东省', '河源': '广东省', '阳江': '广东省', '清远': '广东省', '东莞': '广东省', '中山': '广东省', '潮州': '广东省', '揭阳': '广东省', '云浮': '广东省',
    '广西': '广西壮族自治区', '南宁': '广西壮族自治区', '柳州': '广西壮族自治区', '桂林': '广西壮族自治区', '梧州': '广西壮族自治区', '北海': '广西壮族自治区', '防城港': '广西壮族自治区', '钦州': '广西壮族自治区', '贵港': '广西壮族自治区', '玉林': '广西壮族自治区', '百色': '广西壮族自治区', '贺州': '广西壮族自治区', '河池': '广西壮族自治区', '来宾': '广西壮族自治区', '崇左': '广西壮族自治区',
    '海南': '海南省', '海口': '海南省', '三亚': '海南省', '三沙': '海南省', '儋州': '海南省',
    '四川': '四川省', '成都': '四川省', '自贡': '四川省', '攀枝花': '四川省', '泸州': '四川省', '德阳': '四川省', '绵阳': '四川省', '广元': '四川省', '遂宁': '四川省', '内江': '四川省', '乐山': '四川省', '南充': '四川省', '眉山': '四川省', '宜宾': '四川省', '广安': '四川省', '达州': '四川省', '雅安': '四川省', '巴中': '四川省', '资阳': '四川省', '阿坝': '四川省', '甘孜': '四川省', '凉山': '四川省',
    '贵州': '贵州省', '贵阳': '贵州省', '六盘水': '贵州省', '遵义': '贵州省', '安顺': '贵州省', '毕节': '贵州省', '铜仁': '贵州省', '黔西南': '贵州省', '黔东南': '贵州省', '黔南': '贵州省',
    '云南': '云南省', '昆明': '云南省', '曲靖': '云南省', '玉溪': '云南省', '保山': '云南省', '昭通': '云南省', '丽江': '云南省', '普洱': '云南省', '临沧': '云南省', '楚雄': '云南省', '红河': '云南省', '文山': '云南省', '西双版纳': '云南省', '大理': '云南省', '德宏': '云南省', '怒江': '云南省', '迪庆': '云南省',
    '西藏': '西藏自治区', '拉萨': '西藏自治区', '日喀则': '西藏自治区', '昌都': '西藏自治区', '林芝': '西藏自治区', '山南': '西藏自治区', '那曲': '西藏自治区',
    '陕西': '陕西省', '西安': '陕西省', '铜川': '陕西省', '宝鸡': '陕西省', '咸阳': '陕西省', '渭南': '陕西省', '延安': '陕西省', '汉中': '陕西省', '榆林': '陕西省', '安康': '陕西省', '商洛': '陕西省',
    '甘肃': '甘肃省', '兰州': '甘肃省', '嘉峪关': '甘肃省', '金昌': '甘肃省', '白银': '甘肃省', '天水': '甘肃省', '武威': '甘肃省', '张掖': '甘肃省', '平凉': '甘肃省', '酒泉': '甘肃省', '庆阳': '甘肃省', '定西': '甘肃省', '陇南': '甘肃省', '临夏': '甘肃省', '甘南': '甘肃省',
    '青海': '青海省', '西宁': '青海省', '海东': '青海省', '海北': '青海省', '黄南': '青海省', '海南州': '青海省', '果洛': '青海省', '玉树': '青海省', '海西': '青海省',
    '宁夏': '宁夏回族自治区', '银川': '宁夏回族自治区', '石嘴山': '宁夏回族自治区', '吴忠': '宁夏回族自治区', '固原': '宁夏回族自治区', '中卫': '宁夏回族自治区',
    '新疆': '新疆维吾尔自治区', '乌鲁木齐': '新疆维吾尔自治区', '克拉玛依': '新疆维吾尔自治区', '吐鲁番': '新疆维吾尔自治区', '哈密': '新疆维吾尔自治区', '昌吉': '新疆维吾尔自治区', '博尔塔拉': '新疆维吾尔自治区', '巴音郭楞': '新疆维吾尔自治区', '阿克苏': '新疆维吾尔自治区', '克孜勒苏': '新疆维吾尔自治区', '喀什': '新疆维吾尔自治区', '和田': '新疆维吾尔自治区', '伊犁': '新疆维吾尔自治区',
    '香港': '香港特别行政区', '澳门': '澳门特别行政区',
    '台湾': '台湾省', '台北': '台湾省', '新北': '台湾省', '高雄': '台湾省', '台中': '台湾省', '台南': '台湾省'
};

// 把投稿里的城市或省份名称解析成省级行政区名，用于地图定位
function getProvinceForCity(city) {
    const value = (city || '').trim();
    if (!value) return '';

    if (CITY_PROVINCE_MAP[value]) return CITY_PROVINCE_MAP[value];

    const province = (window.CHINA_PROVINCES || []).find((item) =>
        item.name === value || item.short === value
    );
    return province ? province.name : '';
}

async function getUserPostProvinces() {
    const token = getToken();
    if (!token) return new Set();

    try {
        const res = await fetch(`${API_BASE}/api/my-contributions`, {
            headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) return new Set();

        const posts = await res.json();
        if (!Array.isArray(posts)) return new Set();

        return new Set(
            posts
                .map((post) => getProvinceForCity(post.city))
                .filter(Boolean)
        );
    } catch (err) {
        console.warn('获取投稿省份失败', err);
        return new Set();
    }
}

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
    authBtn: $('auth-btn'),
    authModal: $('auth-modal'),
    authBackdrop: $('auth-backdrop'),
    authClose: $('auth-close'),
    authForm: $('auth-form'),
    authUsername: $('auth-username'),
    authPassword: $('auth-password'),
    authSubmit: $('auth-submit'),
    authStatus: $('auth-status'),
    authTabs: document.querySelectorAll('.auth-tab'),
    sheet: $('history-sheet'),
    sheetBackdrop: $('sheet-backdrop'),
    historyList: $('history-list'),
    tabbar: $('tabbar'),
    communityTitle: $('community-title'),
    communityList: $('community-list'),
    communityRefresh: $('community-refresh'),
    meUserCard: $('me-user-card'),
    meMap: $('me-map'),
    chinaMap: $('china-map'),
    meTabs: document.querySelectorAll('.me-tab'),
    meContent: $('me-content'),
    meLogout: $('me-logout'),
    uploadForm: $('upload-form'),
    uploadGate: $('upload-gate'),
    uploadLoginBtn: $('upload-login-btn'),
    contributeCity: $('contribute-city'),
    contributeTitle: $('contribute-title'),
    contributeContent: $('contribute-content'),
    contributeFile: $('contribute-file'),
    submitContribution: $('submit-contribution'),
    contributeStatus: $('contribute-status'),
};

let currentMethod = 'bm25';
let currentCity = '';
let currentView = 'home';        // 当前激活的页面视图
let currentMeSection = null; // 我的页当前分区（null 时显示中国地图）
let currentMeProvince = '';  // 从地图点击进入时筛选的投稿省份
let abortController = null;
let typingTimer = null;
let toastTimer = null;
let authMode = 'login';
window._cityList = []; // 全局城市名缓存，供 extractCityFromQuestion 使用
const AUTH_TOKEN_KEY = 'travel_qa_token';
const AUTH_USER_KEY = 'travel_qa_user';
const HISTORY_MODE_KEY = 'travel_qa_history_scope';
const FOLLOWS_KEY = 'travel_qa_follows';

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
    setAuthUi();
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
    els.historyClear.addEventListener('click', async () => {
        const token = getToken();
        if (token) {
            try {
                await fetch(`${API_BASE}/api/history`, {
                    method: 'DELETE',
                    headers: { Authorization: `Bearer ${token}` },
                });
            } catch (e) {
                console.warn('清空后端历史失败', e);
            }
        }
        try { localStorage.removeItem(HISTORY_KEY); } catch (e) { /* ignore */ }
        await renderHistory();
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

    // 底部导航栏：页面切换
    els.tabbar.addEventListener('click', (e) => {
        const item = e.target.closest('.tab-item');
        if (!item) return;
        switchView(item.dataset.view);
    });

    // 社区刷新
    els.communityRefresh.addEventListener('click', () => {
        loadCommunityPosts();
        showToast('社区已刷新', 'info');
    });

    // 我的页分区切换
    els.meTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const section = tab.dataset.section || 'history';
            // 通过底部 tab 切换时取消地图省份筛选，避免残留到后面的“我的投稿”视图
            currentMeProvince = '';
            // 再次点击当前分区时取消选择，回到中国地图
            currentMeSection = currentMeSection === section ? null : section;
            showMeSection();
        });
    });
    els.meLogout.addEventListener('click', logoutUser);

    // 上传页：未登录引导 + 表单提交
    els.uploadLoginBtn.addEventListener('click', () => {
        openAuthModal();
        setAuthStatus('请先登录后再提交旅游经验', 'warn');
    });
    els.uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        submitContribution();
    });
    els.contributeFile.addEventListener('change', () => {
        const file = els.contributeFile.files && els.contributeFile.files[0];
        const label = document.getElementById('upload-file-name');
        if (label) label.textContent = file ? file.name : '上传文件';
    });

    // 社区列表内：关注 / 取消关注
    els.communityList.addEventListener('click', async (e) => {
        const followBtn = e.target.closest('.follow-btn');
        if (followBtn) {
            await toggleFollowByName(followBtn.dataset.username, followBtn);
            return;
        }
        const viewBtn = e.target.closest('.follow-view');
        if (viewBtn) {
            switchView('community', { skipLoad: true });
            await loadCommunityPosts(viewBtn.dataset.username);
            return;
        }
    });

    // 我的页内容内：查看 TA 的帖子 / 取消关注
    els.meContent.addEventListener('click', async (e) => {
        const viewBtn = e.target.closest('.follow-view');
        if (viewBtn) {
            switchView('community', { skipLoad: true });
            await loadCommunityPosts(viewBtn.dataset.username);
            return;
        }
        const unBtn = e.target.closest('.follow-un');
        if (unBtn) {
            removeFollow(unBtn.dataset.username);
            renderFollows();
            return;
        }
    });

    els.authBtn.addEventListener('click', () => {
        const token = localStorage.getItem(AUTH_TOKEN_KEY);
        if (token) {
            logoutUser();
            return;
        }
        openAuthModal();
    });

    els.authBackdrop.addEventListener('click', closeAuthModal);
    els.authClose.addEventListener('click', closeAuthModal);
    els.authTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            authMode = tab.dataset.mode || 'login';
            els.authTabs.forEach(item => item.classList.toggle('active', item === tab));
            const isLogin = authMode === 'login';
            els.authSubmit.textContent = isLogin ? '登录' : '注册';
            setAuthStatus('');
        });
    });
    els.authForm.addEventListener('submit', handleAuthSubmit);
}

function runQuickQuery(q) {
    els.questionInput.value = q;
    const city = extractCityFromQuestion(q);
    if (city) setCityFilter(city);
    askQuestion();
}

function getToken() {
    return localStorage.getItem(AUTH_TOKEN_KEY) || '';
}

function getCurrentUser() {
    try {
        return JSON.parse(localStorage.getItem(AUTH_USER_KEY) || 'null');
    } catch (e) {
        return null;
    }
}

function setAuthUi() {
    const user = getCurrentUser();
    const token = getToken();
    if (user && token) {
        els.authBtn.textContent = `已登录 · ${user.username}`;
        els.authBtn.title = `已登录：${user.username}`;
        localStorage.setItem(HISTORY_MODE_KEY, 'user');
    } else {
        els.authBtn.textContent = '登录';
        els.authBtn.title = '登录/注册';
        localStorage.setItem(HISTORY_MODE_KEY, 'guest');
    }
    // 联动刷新当前视图（我的 / 上传经验）
    if (currentView === 'me') renderMeView();
    if (currentView === 'upload') renderUploadView();
}

function openAuthModal() {
    els.authModal.hidden = false;
    els.authUsername.focus();
}

function closeAuthModal() {
    els.authModal.hidden = true;
    els.authForm.reset();
    setAuthStatus('');
}

function setAuthStatus(message, type = '') {
    els.authStatus.textContent = message;
    els.authStatus.className = 'auth-status' + (type ? ` ${type}` : '');
}

function formatApiError(data, fallback = '认证失败') {
    const detail = data && data.detail;
    if (typeof detail === 'string' && detail.trim()) return detail;
    if (Array.isArray(detail)) {
        const messages = detail
            .map(item => typeof item === 'string' ? item : item && item.msg)
            .filter(Boolean);
        if (messages.length) return messages.join('；');
    }
    if (data && typeof data.message === 'string' && data.message.trim()) {
        return data.message;
    }
    return fallback;
}

async function handleAuthSubmit(event) {
    event.preventDefault();
    const username = els.authUsername.value.trim();
    const password = els.authPassword.value.trim();

    if (!username || !password) {
        setAuthStatus('请填写用户名和密码', 'error');
        return;
    }

    els.authSubmit.disabled = true;
    setAuthStatus(authMode === 'login' ? '正在登录…' : '正在注册…', '');

    try {
        const endpoint = authMode === 'login' ? '/api/login' : '/api/register';
        const res = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            throw new Error(formatApiError(data));
        }

        localStorage.setItem(AUTH_TOKEN_KEY, data.token || '');
        localStorage.setItem(AUTH_USER_KEY, JSON.stringify(data.user || { username }));
        setAuthUi();
        closeAuthModal();
        showToast(authMode === 'login' ? '登录成功' : '注册成功', 'ok');
    } catch (err) {
        setAuthStatus(err.message || '认证失败', 'error');
    } finally {
        els.authSubmit.disabled = false;
    }
}

function logoutUser() {
    localStorage.removeItem(AUTH_TOKEN_KEY);
    localStorage.removeItem(AUTH_USER_KEY);
    localStorage.removeItem(HISTORY_KEY);
    setAuthUi();
    renderHistory();
    showToast('已退出登录，历史已清空', 'info');
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
            const title = escapeHtml(src.title || '');
            const sourceLabel = escapeHtml(src.source || '来源');
            const url = (src.source_url || src.sourceUrl || '').trim();
            const sourceId = (src.submission_id || src.id || '').trim();
            const username = (src.username || '').trim();
            const isCommunitySource = (src.source === '用户亲身经历' || !!src.username || !!src.user_id || !!sourceId);
            const sourceText = isCommunitySource
                ? `<button type="button" class="src-link source-link" data-source-id="${escapeHtml(sourceId)}" data-username="${escapeHtml(username)}" title="${sourceLabel}">${title}</button>`
                : url
                    ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" class="src-link" title="${sourceLabel}">${title}</a>`
                    : `<span class="src-title">${title}</span>`;
            const tag = src.source ? `<span class="src-source-tag" title="${sourceLabel}">${sourceLabel}</span>` : '';
            sources += `<span class="src-chip"><i>${idx + 1}</i>${sourceText}${tag}</span>`;
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
    const sourceLinks = wrap.querySelectorAll('.source-link');
    sourceLinks.forEach((button) => {
        button.addEventListener('click', async () => {
            const sourceId = (button.dataset.sourceId || '').trim();
            const username = (button.dataset.username || '').trim();
            const sourceTitle = button.textContent.trim();
            const sourceCity = button.closest('.src-chip')?.parentElement?.dataset?.city || '';
            switchView('community', { skipLoad: true });
            if (!sourceId && username) {
                await loadCommunityPosts(username);
                return;
            }
            await openCommunityPostDetail(sourceId, username, sourceTitle, sourceCity);
        });
    });
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

    const token = getToken();
    if (!token) {
        openAuthModal();
        setAuthStatus('请先登录后再提交旅游经验', 'warn');
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
            headers: { Authorization: `Bearer ${token}` },
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
            const fileLabel = document.getElementById('upload-file-name');
            if (fileLabel) fileLabel.textContent = '上传文件';
            // 若正停留在「我的 - 投稿」分区，刷新列表
            if (currentView === 'me' && currentMeSection === 'posts') {
                await renderMeSection();
            }
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

// ========== 页面视图切换 ==========
function switchView(name, opts = {}) {
    currentView = name || 'home';
    // 导航栏激活态
    els.tabbar.querySelectorAll('.tab-item').forEach(b => {
        b.classList.toggle('active', b.dataset.view === currentView);
    });
    // 视图激活态
    document.querySelectorAll('.view').forEach(v => {
        v.classList.toggle('active', v.dataset.view === currentView);
    });
    closeSheet();
    hideToast();
    // 懒加载各页数据（skipLoad 用于需要自行加载数据的场景，避免竞态）
    if (opts.skipLoad) return;
    if (currentView === 'community') {
        loadCommunityPosts();
    } else if (currentView === 'upload') {
        renderUploadView();
    } else if (currentView === 'me') {
        renderMeView();
    }
}

// ========== 上传经验页 ==========
function renderUploadView() {
    const loggedIn = !!getToken();
    els.uploadForm.hidden = !loggedIn;
    els.uploadGate.hidden = loggedIn;
    if (loggedIn) els.contributeStatus.textContent = '';
}

// ========== 我的页 ==========
async function renderMeView() {
    const user = getCurrentUser();
    const token = getToken();
    if (user && token) {
        const initial = (user.username || '我').trim().charAt(0).toUpperCase();
        els.meUserCard.innerHTML = `
            <div class="me-avatar">${escapeHtml(initial)}</div>
            <div class="me-info">
                <div class="me-name">${escapeHtml(user.username)}</div>
                <div class="me-sub">已登录 · 历史与投稿已同步到云端</div>
            </div>`;
        els.meLogout.hidden = false;
    } else {
        els.meUserCard.innerHTML = `
            <div class="me-avatar">👤</div>
            <div class="me-info">
                <div class="me-name">未登录</div>
                <div class="me-sub">登录后体验投稿、历史同步等功能</div>
            </div>
            <button type="button" class="primary-btn me-login-btn" id="me-login-btn">立即登录 / 注册</button>`;
        els.meLogout.hidden = true;
        const loginBtn = document.getElementById('me-login-btn');
        if (loginBtn) loginBtn.addEventListener('click', () => openAuthModal());
    }
    showMeSection();
}

async function showMeSection() {
    const hasSection = currentMeSection !== null;
    els.meMap.hidden = hasSection;
    els.meContent.hidden = !hasSection;
    els.meTabs.forEach(tab => tab.classList.toggle('active', tab.dataset.section === currentMeSection));

    if (hasSection) {
        await renderMeSection();
    } else {
        currentMeProvince = '';
        await renderChinaMap();
    }
}

async function renderChinaMap() {
    const svg = els.chinaMap;
    if (!svg || !window.CHINA_PROVINCES || !window.CHINA_PROVINCES.length) return;

    const postProvinces = await getUserPostProvinces();
    const paths = window.CHINA_PROVINCES.map((province) => {
        const hasUserPost = postProvinces.has(province.name);
        const classes = [
            'map-province',
            province.covered ? 'covered' : '',
            hasUserPost ? 'has-user-post' : ''
        ].filter(Boolean).join(' ');
        return `<path class="${classes}" d="${province.path}" data-name="${escapeHtml(province.name)}"></path>`;
    }).join('');

    svg.innerHTML = `${paths}<text class="china-map-tooltip" x="500" y="80" text-anchor="middle"></text>`;

    const tooltip = svg.querySelector('.china-map-tooltip');
    svg.querySelectorAll('path.map-province').forEach((path) => {
        path.addEventListener('mouseenter', () => {
            tooltip.textContent = path.dataset.name || '';
        });
        path.addEventListener('mouseleave', () => {
            tooltip.textContent = '';
        });
        path.addEventListener('click', () => {
            const provinceName = path.dataset.name || '';
            if (!path.classList.contains('has-user-post') || !provinceName) return;

            currentMeProvince = provinceName;
            currentMeSection = 'posts';
            showMeSection();
        });
    });
}

function renderMeSection() {
    if (currentMeSection === 'history') {
        renderHistoryInto(els.meContent, { navigateHome: true });
    } else if (currentMeSection === 'follows') {
        renderFollows();
    } else if (currentMeSection === 'posts') {
        if (!getToken()) {
            els.meContent.innerHTML = '<div class="home-empty">登录后即可查看自己的投稿。</div>';
            return;
        }
        loadUserHomePosts(els.meContent, currentMeProvince);
    }
}

// ========== 关注列表（localStorage 本地实现） ==========
function getFollows() {
    try {
        const list = JSON.parse(localStorage.getItem(FOLLOWS_KEY) || '[]');
        return Array.isArray(list) ? list : [];
    } catch (e) {
        return [];
    }
}

function saveFollows(list) {
    try { localStorage.setItem(FOLLOWS_KEY, JSON.stringify(list)); } catch (e) { /* ignore */ }
}

function isFollowing(username) {
    const name = (username || '').trim();
    if (!name) return false;
    return getFollows().some(f => f.username === name);
}

async function toggleFollowByName(username, btn) {
    const name = (username || '').trim();
    if (!name) return;
    let list = getFollows();
    if (isFollowing(name)) {
        list = list.filter(f => f.username !== name);
        saveFollows(list);
        if (btn) {
            btn.textContent = '关注';
            btn.classList.remove('following');
            btn.title = '关注该作者';
        }
        showToast(`已取消关注 ${name}`, 'info');
    } else {
        list.push({ username: name, followedAt: new Date().toISOString() });
        saveFollows(list);
        if (btn) {
            btn.textContent = '已关注';
            btn.classList.add('following');
            btn.title = '点击取消关注';
        }
        showToast(`已关注 ${name}`, 'ok');
    }
}

function removeFollow(username) {
    const name = (username || '').trim();
    if (!name) return;
    saveFollows(getFollows().filter(f => f.username !== name));
    showToast(`已取消关注 ${name}`, 'info');
}

function renderFollows() {
    const list = getFollows();
    if (!list.length) {
        els.meContent.innerHTML = `
            <div class="home-empty">
                还没有关注任何人<br>
                去 <button type="button" class="inline-link" id="go-community">社区</button> 看看有趣的旅行者吧
            </div>`;
        const go = document.getElementById('go-community');
        if (go) go.addEventListener('click', () => switchView('community'));
        return;
    }
    els.meContent.innerHTML = list.map(f => `
        <div class="follow-item">
            <div class="follow-avatar">${escapeHtml((f.username || '?').charAt(0).toUpperCase())}</div>
            <div class="follow-info">
                <div class="follow-name">${escapeHtml(f.username)}</div>
                <div class="follow-sub">关注于 ${formatTime(f.followedAt)}</div>
            </div>
            <div class="follow-actions">
                <button type="button" class="follow-view" data-username="${escapeHtml(f.username)}">看TA的帖子</button>
                <button type="button" class="follow-un" data-username="${escapeHtml(f.username)}">取消</button>
            </div>
        </div>
    `).join('');
}

async function loadCommunityPosts(username = '') {
    try {
        const url = username
            ? `${API_BASE}/api/community?username=${encodeURIComponent(username)}`
            : `${API_BASE}/api/community`;
        const res = await fetch(url);
        if (!res.ok) throw new Error('获取社区帖子失败');
        const posts = await res.json();
        if (!posts.length) {
            els.communityList.innerHTML = '<div class="home-empty">社区里还没有帖子，快来发布第一篇吧。</div>';
            return;
        }

        const title = username ? `${username} 的社区主页` : '用户动态';
        els.communityTitle.textContent = title;
        els.communityList.innerHTML = posts.map((post) => {
            const author = (post.username || '').trim();
            const followBtn = author
                ? `<button type="button" class="follow-btn${isFollowing(author) ? ' following' : ''}" data-username="${escapeHtml(author)}" title="${isFollowing(author) ? '点击取消关注' : '关注该作者'}">${isFollowing(author) ? '已关注' : '关注'}</button>`
                : '';
            return `
            <article class="user-post-card community-post-card" data-post-id="${escapeHtml(post.id || '')}">
                <div class="post-topline">
                    <span class="post-city">📍 ${escapeHtml(post.city || '未知城市')}</span>
                    <span class="post-time">${formatTime(post.created_at || post.updated_at)}</span>
                </div>
                <div class="community-author-line">
                    <span>作者：${escapeHtml(author || '用户')}</span>
                    ${followBtn}
                </div>
                <h4>${escapeHtml(post.title || '旅游心得')}</h4>
                <p>${escapeHtml((post.content || '').slice(0, 180))}${(post.content || '').length > 180 ? '…' : ''}</p>
                <div class="post-meta">
                    <span>${escapeHtml(post.source || '用户亲身经历')}</span>
                    <span>${escapeHtml(post.username || '用户')}</span>
                </div>
            </article>
        `}).join('');

        els.communityList.querySelectorAll('.community-post-card').forEach((card) => {
            card.addEventListener('click', async () => {
                const postId = card.dataset.postId;
                if (!postId) return;
                await openCommunityPostDetail(postId);
            });
        });
    } catch (err) {
        console.warn(err);
        els.communityList.innerHTML = '<div class="home-empty">社区帖子加载失败，请稍后重试。</div>';
    }
}

async function findCommunityPostBySource(sourceTitle, sourceCity = '') {
    const title = (sourceTitle || '').trim();
    const city = (sourceCity || '').trim();
    if (!title) return null;

    try {
        const res = await fetch(`${API_BASE}/api/community`);
        if (!res.ok) return null;
        const posts = await res.json();
        const idByExactMatch = posts.find((post) => {
            const sameTitle = (post.title || '').trim() === title;
            const sameCity = !city || (post.city || '').trim() === city;
            return sameTitle && sameCity;
        });
        if (idByExactMatch) return idByExactMatch.id || null;

        const idByTitle = posts.find((post) => (post.title || '').trim() === title);
        return idByTitle ? (idByTitle.id || null) : null;
    } catch (err) {
        console.warn('查找社区对应帖子失败', err);
        return null;
    }
}

async function openCommunityPostDetail(postId, username = '', sourceTitle = '', sourceCity = '') {
    const authorName = (username || '').trim();
    let resolvedId = (postId || '').trim();

    if (!resolvedId) {
        resolvedId = await findCommunityPostBySource(sourceTitle, sourceCity);
    }

    if (!resolvedId) {
        if (authorName) {
            await loadCommunityPosts(authorName);
        } else {
            switchView('community', { skipLoad: true });
            els.communityTitle.textContent = '用户动态';
            await loadCommunityPosts();
        }
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/api/community/${resolvedId}`);
        if (!res.ok) {
            const fallbackId = await findCommunityPostBySource(sourceTitle, sourceCity);
            if (fallbackId) {
                await openCommunityPostDetail(fallbackId, authorName, sourceTitle, sourceCity);
                return;
            }
            if (authorName) {
                await loadCommunityPosts(authorName);
                return;
            }
            throw new Error('社区帖子获取失败');
        }
        const post = await res.json();
        const displayAuthor = (post.username || authorName || '').trim();
        const followBtn = displayAuthor
            ? `<button type="button" class="follow-btn${isFollowing(displayAuthor) ? ' following' : ''}" data-username="${escapeHtml(displayAuthor)}" title="${isFollowing(displayAuthor) ? '点击取消关注' : '关注该作者'}">${isFollowing(displayAuthor) ? '已关注' : '关注'}</button>`
            : '';
        els.communityTitle.textContent = `${displayAuthor || '用户'} 的社区主页`;
        els.communityList.innerHTML = `
            <div class="user-post-detail community-post-detail">
                <button type="button" class="back-link" data-action="back-to-community">← 返回社区</button>
                <div class="post-detail-header">
                    <span class="post-city">📍 ${escapeHtml(post.city || '未知城市')}</span>
                    <span class="post-time">${formatTime(post.created_at || post.updated_at)}</span>
                </div>
                <div class="community-author-line">
                    <span>作者：${escapeHtml(displayAuthor || '用户')}</span>
                    ${followBtn}
                </div>
                <h4>${escapeHtml(post.title || '旅游心得')}</h4>
                <div class="post-detail-source">${escapeHtml(post.source || '用户亲身经历')}</div>
                <div class="post-detail-body">${escapeHtml(post.content || '').replace(/\n/g, '<br>')}</div>
            </div>
        `;

        const backBtn = els.communityList.querySelector('[data-action="back-to-community"]');
        if (backBtn) {
            backBtn.addEventListener('click', () => loadCommunityPosts(displayAuthor));
        }
    } catch (err) {
        console.warn(err);
        switchView('community', { skipLoad: true });
        els.communityTitle.textContent = '用户动态';
        await loadCommunityPosts();
    }
}

async function loadUserHomePosts(container = els.meContent, province = '') {
    const token = getToken();
    if (!token) return;
    if (!container) return;

    try {
        const res = await fetch(`${API_BASE}/api/my-contributions`, {
            headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) {
            if (res.status === 401) {
                logoutUser();
                return;
            }
            throw new Error('获取帖子列表失败');
        }
        const posts = await res.json();

        const listPosts = Array.isArray(posts) ? posts : [];
        const filtered = province
            ? listPosts.filter((post) => getProvinceForCity(post.city) === province)
            : listPosts;

        const backButton = province
            ? `<button type="button" class="map-back-link" data-action="back-to-map">← 返回地图</button>`
            : '';
        const listHeader = province
            ? `<div class="province-posts-head">${escapeHtml(province)}的投稿</div>`
            : '';

        if (!filtered.length) {
            const emptyText = province
                ? `还没有发布过${escapeHtml(province)}的旅游经验。`
                : '还没有发布过心得，快去上传经验吧。';
            container.innerHTML = `
                ${backButton}${listHeader}
                <div class="home-empty">${emptyText}</div>
            `;
            bindPostsBackToMap(container);
            return;
        }

        container.innerHTML = `
            ${backButton}${listHeader}
            ${filtered.map((post) => `
            <article class="user-post-card" data-post-id="${escapeHtml(post.id || '')}">
                <div class="post-topline">
                    <span class="post-city">📍 ${escapeHtml(post.city || '未知城市')}</span>
                    <span class="post-time">${formatTime(post.created_at || post.updated_at)}</span>
                </div>
                <h4>${escapeHtml(post.title || '旅游心得')}</h4>
                <p>${escapeHtml((post.content || '').slice(0, 180))}${(post.content || '').length > 180 ? '…' : ''}</p>
                <div class="post-meta">
                    <span>${escapeHtml(post.source || '用户亲身经历')}</span>
                    <span>${escapeHtml(post.username || '用户')}</span>
                </div>
            </article>
            `).join('')}
        `;

        container.querySelectorAll('.user-post-card').forEach((card) => {
            card.addEventListener('click', async () => {
                const postId = card.dataset.postId;
                if (!postId) return;
                await openUserPostDetail(postId, container, province);
            });
        });

        bindPostsBackToMap(container);
    } catch (err) {
        container.innerHTML = '<div class="home-empty">帖子加载失败，请稍后重试。</div>';
        console.warn(err);
    }
}

function bindPostsBackToMap(container) {
    const backBtn = container.querySelector('[data-action="back-to-map"]');
    if (!backBtn) return;
    backBtn.addEventListener('click', () => {
        currentMeSection = null;
        currentMeProvince = '';
        showMeSection();
    });
}

async function openUserPostDetail(postId, container = els.meContent, province = '') {
    const token = getToken();
    if (!token) return;
    if (!container) return;

    try {
        const res = await fetch(`${API_BASE}/api/my-contributions/${postId}`, {
            headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) {
            if (res.status === 401) {
                logoutUser();
                return;
            }
            throw new Error('获取帖子详情失败');
        }
        const post = await res.json();
        container.innerHTML = `
            <div class="user-post-detail">
                <button type="button" class="map-back-link" data-action="back-to-list">← 返回${province ? escapeHtml(province) + '投稿' : '我的投稿'}</button>
                <div class="post-detail-header">
                    <span class="post-city">📍 ${escapeHtml(post.city || '未知城市')}</span>
                    <span class="post-time">${formatTime(post.created_at || post.updated_at)}</span>
                </div>
                <h4>${escapeHtml(post.title || '旅游心得')}</h4>
                <div class="post-detail-source">${escapeHtml(post.source || '用户亲身经历')}</div>
                <div class="post-detail-body">${escapeHtml(post.content || '').replace(/\n/g, '<br>')}</div>
                <div class="post-detail-actions">
                    <button type="button" class="danger-btn" data-action="delete-post" data-post-id="${escapeHtml(post.id || '')}">删除帖子</button>
                </div>
            </div>
        `;

        const backBtn = container.querySelector('[data-action="back-to-list"]');
        if (backBtn) {
            backBtn.addEventListener('click', () => loadUserHomePosts(container, province));
        }

        const deleteBtn = container.querySelector('[data-action="delete-post"]');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', async () => {
                const confirmDelete = window.confirm('确定删除这篇帖子吗？');
                if (!confirmDelete) return;
                const deleteRes = await fetch(`${API_BASE}/api/my-contributions/${postId}`, {
                    method: 'DELETE',
                    headers: { Authorization: `Bearer ${token}` },
                });
                if (!deleteRes.ok) {
                    showToast('删除失败', 'error');
                    return;
                }
                showToast('帖子已删除', 'ok');
                await loadUserHomePosts(container, province);
            });
        }
    } catch (err) {
        console.warn(err);
        showToast('帖子详情加载失败', 'error');
    }
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

// ========== 问答历史 (localStorage) ==========
async function getHistory() {
    const token = getToken();
    if (token) {
        try {
            const res = await fetch(`${API_BASE}/api/history`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            if (!res.ok) {
                if (res.status === 401) {
                    logoutUser();
                    return [];
                }
                return [];
            }
            return await res.json();
        } catch (e) {
            return [];
        }
    }

    try {
        const raw = localStorage.getItem(HISTORY_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch (e) {
        return [];
    }
}

async function saveToHistory(question, data) {
    const token = getToken();
    const payload = {
        question: question,
        answer: data.answer ? data.answer.substring(0, 100) + '...' : '',
        detected_city: data.detected_city,
        timestamp: new Date().toISOString(),
    };

    if (token) {
        try {
            const res = await fetch(`${API_BASE}/api/history`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${token}`,
                },
                body: JSON.stringify(payload),
            });
            if (res.ok) {
                localStorage.removeItem(HISTORY_KEY);
                return;
            }
        } catch (e) {
            console.warn('记录历史失败，回退到本地存储', e);
        }
    }

    const history = await getHistory();
    history.unshift({
        question: question,
        answer: data.answer ? data.answer.substring(0, 100) + '...' : '',
        detected_city: data.detected_city,
        timestamp: new Date().toISOString(),
        fullData: data,
    });
    if (history.length > MAX_HISTORY) history.length = MAX_HISTORY;
    try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch (e) {
        history.length = 10;
        localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    }
}

async function renderHistoryInto(container, opts = {}) {
    const history = await getHistory();
    if (!container) return;
    if (history.length === 0) {
        container.innerHTML = '<div class="h-empty">暂无问答记录</div>';
        return;
    }
    container.innerHTML = history.slice(0, 10).map((item, idx) => `
            <div class="history-item" data-idx="${idx}">
                <div class="h-item-q">${escapeHtml(item.question)}</div>
                <div class="h-item-meta">${item.detected_city ? '📍 ' + escapeHtml(item.detected_city) + ' · ' : ''}${formatTime(item.timestamp)}</div>
            </div>
        `).join('');

    // 点击历史记录恢复对话
    container.querySelectorAll('.history-item').forEach(el => {
        el.addEventListener('click', () => {
            const idx = parseInt(el.dataset.idx);
            const item = history[idx];
            if (item && item.fullData) {
                els.questionInput.value = item.question;
                if (item.fullData.detected_city) {
                    setCityFilter(item.fullData.detected_city);
                }
                closeSheet();
                // 从其它页面恢复时回到主页
                if (opts.navigateHome || currentView !== 'home') {
                    switchView('home', { skipLoad: true });
                }
                pushUserMessage(item.question);
                pushAiMessage(item.fullData);
            }
        });
    });
}

async function renderHistory() {
    await renderHistoryInto(els.historyList, { navigateHome: true });
}

async function loadHistory() {
    await renderHistory();
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
