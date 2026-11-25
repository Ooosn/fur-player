// What the page shows and what it is called. Change names, order and defaults here, not in the code.

export const MOTIONS = [
  // Free (v23, the author 11-25: the dog keeps moving, choosing freely among the motions): FREE below, renderer.loadFree
  { id: 'free', free: true, zh: '自由', en: 'Free', key: '0' },
  { id: 'walk', clip: 'mocap_walk', zh: '散步', en: 'Walk', key: '1' },
  { id: 'trot', clip: 'mocap_trot_leap', zh: '小跑', en: 'Trot', key: '2' },
  { id: 'gallop', clip: 'mocap_gallop_turn', zh: '奔跑', en: 'Gallop', key: '3' },
  { id: 'jump', clip: 'mocap_high_jump', zh: '跳跃', en: 'Jump', key: '4' },
  { id: 'sit', clip: 'mocap_sit_stand', zh: '坐下', en: 'Sit', key: '5' },
  { id: 'shake', clip: 'shake_strong', zh: '抖毛', en: 'Shake', key: '6' },
];
export const DEFAULT_MOTION = 'free';
// the clips the Free motion chains (weights = how often each is picked; never the same twice in a row), the length of
// one random sequence (a new one is drawn when it ends) and the cross-fade between clips (frames at 30 fps)
export const FREE = {
  seconds: 180, fade: 12,
  clips: [
    { clip: 'mocap_trot_leap', w: 3 }, { clip: 'mocap_walk', w: 2 }, { clip: 'mocap_gallop_turn', w: 2 },
    { clip: 'mocap_high_jump', w: 1 }, { clip: 'mocap_sit_stand', w: 1 }, { clip: 'shake_strong', w: 1.2 },
  ],
};

// views: what each half of the screen draws
export const VIEWS = {
  fur: { zh: '毛发渲染', en: 'Fur', style: 0, colour: 0, body: 1, fur: true, kernels: false },
  gauss: { zh: '高斯', en: 'Gaussians', style: 3, colour: 5, body: 2, fur: true, kernels: false },
  kernels: { zh: '场的核', en: 'Kernels', style: 0, colour: 0, body: 2, fur: false, kernels: true },
};
export const DEFAULT_VIEW = 'fur';
export const DEFAULT_COMPARE = false;       // divider: fur on the left, the chosen other view on the right
export const DEFAULT_SPLIT = 0.55;

// Gaussian view defaults (the toolbar of that view changes them)
export const GAUSS = { size: 1.0, density: 0.04, colour: 5, style: 1 };
export const GAUSS_COLOURS = [
  { v: 5, zh: '每根毛一种颜色', en: 'One colour per strand' },
  { v: 0, zh: '原生颜色', en: 'Native colour' },
  { v: 2, zh: '方向 0.5(t+1)', en: 'Direction 0.5(t+1)' },
  { v: 3, zh: '弯曲角', en: 'Bend angle' },
];
export const GAUSS_STYLES = [
  { v: 1, zh: '轮廓（2σ）', en: 'Outlines (2σ)' },
  { v: 3, zh: '实心椭圆', en: 'Solid ellipses' },
  { v: 0, zh: '原样', en: 'As rendered' },
];

// dynamics presets shown by name (keys of the manifest's presets)
export const PRESETS = [
  { id: 'accepted', zh: '标准', en: 'Standard' },
  { id: 'lively_20261119', zh: '活泼', en: 'Lively' },
  { id: 'lively_soft_20261119', zh: '柔软', en: 'Soft' },
];
export const DEFAULT_PRESET = 'lively_soft_20261119';

// how strands move (v20, the author 11-25: "两个版本都做，在设置里面可以选择"): 0 = the kernels move every strand (as before);
// 1 = kernels + each strand answers the hand by itself, and the kernels' motion is decorrelated per strand;
// 2 = strands only, no kernels: one oscillator per strand, neighbours coupled by velocity friction (shaders.js SDYN)
export const STRAND_MODES = [
  { id: 0, zh: '场的核', en: 'Kernels', short: { zh: '场的核', en: 'Kernels' },
    note: { zh: '和原来一样：每根毛都跟着 2 万个场核一起动。', en: 'As before: every strand follows the 20,000 field kernels.' } },
  { id: 1, zh: '核 + 逐根', en: 'Kernels + strands', short: { zh: '核 + 逐根', en: 'Both' },
    note: { zh: '场核带动整片毛；手和细小的时间差让每根毛各自动。', en: 'Kernels move the coat; the hand and small timing differences move each strand on its own.' } },
  { id: 2, zh: '仅逐根', en: 'Strands only', short: { zh: '仅逐根', en: 'Strands' },
    note: { zh: '不用场核：每根毛自己动，相邻的毛互相牵连（见“邻毛牵连”）。', en: 'No kernels: every strand moves by itself; neighbours pull each other (Strand cohesion).' } },
];
export const DEFAULT_STRAND_MODE = 1;
export const STRAND_SLIDERS = [
  // scales the per-strand jitter of frequency, damping, gain, lag, blend weights and how the hand catches each strand
  { key: 'variation', zh: '逐根差异', en: 'Strand variation', unit: '', min: 0, max: 2, step: 0.05, value: 1 },
  // mode 2: velocity friction toward the strands around (per second); high = locks, low = single strands
  { key: 'coh', zh: '邻毛牵连', en: 'Strand cohesion', unit: '/s', min: 0, max: 100, step: 1, value: 25, mode: 2 },
];

export const SLIDERS = [
  { key: 'f_ref', zh: '刚度', en: 'Stiffness', unit: 'Hz', min: 1, max: 6, step: 0.1 },
  // the longest fur (tail, ruff, belly fringe) sits at this floor of the natural frequency: f_ref does not reach it
  { key: 'f_min', zh: '长毛刚度下限', en: 'Long-fur stiffness floor', unit: 'Hz', min: 0.5, max: 3, step: 0.1 },
  { key: 'zeta', zh: '阻尼比', en: 'Damping ratio', unit: '', min: 0, max: 0.9, step: 0.01 },
  { key: 'drag', zh: '空气阻力', en: 'Air drag', unit: '', min: 0, max: 8, step: 0.1 },
  { key: 'cohesion', zh: '成簇程度', en: 'Cohesion', unit: '', min: 0, max: 150, step: 1 },
  { key: 'floor_give_deg', zh: '下垂余量', en: 'Floor give', unit: '°', min: 0, max: 20, step: 0.5 },
  { key: 'inertia_gain', zh: '惯性', en: 'Inertia', unit: '', min: 0, max: 2, step: 0.05 },
  // body accelerations above this are clipped before they drive the fur (running and jumping exceed 2 g)
  { key: 'inertia_limit_g', zh: '惯性上限', en: 'Inertia cap', unit: 'g', min: 0.5, max: 8, step: 0.5 },
  // the strand's bend is saturated at this angle when it is rendered
  { key: 'sat_deg', zh: '最大弯曲', en: 'Max bend', unit: '°', min: 10, max: 60, step: 1 },
  { key: 'gravity_gain', zh: '重力', en: 'Gravity', unit: '', min: 0, max: 2, step: 0.05 },
];

// loose skin of the trunk (skin.js); applied when a clip is loaded, so these act when the slider is released
export const SKIN_SLIDERS = [
  { key: 'amount', zh: '松弛程度', en: 'Looseness', unit: '×', min: 0, max: 2, step: 0.05 },
  { key: 'hz', zh: '皮肤回弹频率', en: 'Skin spring', unit: 'Hz', min: 2, max: 8, step: 0.1 },
];

// About: only the required citation, the licences and the controls (the author, 11-20)
export const ABOUT = {
  // our paper only (the author, 11-20); anonymous while under review
  cite: { authors: 'Anonymous', title: 'Fur reconstruction from images',
    venue: { zh: '审稿中', en: 'Under review' } },
  // the motion capture's credit stays here: CC BY-NC requires naming its creators
  licences: [
    { what: { zh: '本项目的毛发重建', en: 'Fur reconstruction (this project)' }, lic: 'Apache-2.0' },     // the author's choice, 11-20
    { what: { zh: '狗的动作捕捉：Zhang, Starke, Komura, Saito 2018（AI4Animation）', en: 'Dog motion capture: Zhang, Starke, Komura, Saito 2018 (AI4Animation)' },
      lic: 'CC BY-NC 4.0', note: { zh: '非商业用途', en: 'non-commercial' } },
    { what: { zh: '高斯投影（移植自 gsplat）', en: 'Gaussian projection (ported from gsplat)' }, lic: 'Apache-2.0' },
  ],
  // [X] is drawn as a key, the rest as plain text; ' / ' separates alternatives
  controls: [
    [{ zh: '拖动', en: 'Drag' }, { zh: '旋转', en: 'Orbit' }],
    [{ zh: '右键拖动 / [Shift] + 拖动', en: 'Right-drag / [Shift] + drag' }, { zh: '平移', en: 'Pan' }],
    [{ zh: '滚轮', en: 'Wheel' }, { zh: '缩放', en: 'Zoom' }],
    [{ zh: '双击 / [R]', en: 'Double-click / [R]' }, { zh: '复位视角', en: 'Reset view' }],
    [{ zh: '单击高斯', en: 'Click a Gaussian' }, { zh: '查看它属于哪根毛', en: 'Show its strand' }],
    [{ zh: '[空格]', en: '[Space]' }, { zh: '播放 / 暂停', en: 'Play / pause' }],
    [{ zh: '[0] – [6]', en: '[0] – [6]' }, { zh: '动作（0 = 自由）', en: 'Motions (0 = Free)' }],
    [{ zh: '[S]', en: '[S]' }, { zh: '慢放', en: 'Slow motion' }],
    [{ zh: '[D]', en: '[D]' }, { zh: '毛发动力学', en: 'Fur dynamics' }],
    [{ zh: '[C]', en: '[C]' }, { zh: '对比', en: 'Compare' }],
    [{ zh: '[G]', en: '[G]' }, { zh: '高斯视图', en: 'Gaussians' }],
    [{ zh: '[F]', en: '[F]' }, { zh: '全屏', en: 'Full screen' }],
    [{ zh: '[H]', en: '[H]' }, { zh: '抚摸', en: 'Stroke' }],
    [{ zh: '[Esc]', en: '[Esc]' }, { zh: '退出互动', en: 'Leave the tool' }],
  ],
};

// white page (the author, 11-25: "背景改成白色试试"); v17 had a warm grey gradient, top [0.965, 0.963, 0.955],
// bottom [0.845, 0.842, 0.830], vignette 1
export const BG = { top: [1, 1, 1], bottom: [1, 1, 1], vignette: 0 };

// colour of the body mesh under the fur (the author, 11-25): baked from the fur colours (as before), one plain colour
// (default a neutral grey; a skin tone as the second swatch; any colour with the picker), or none (page colour)
export const BODY_MODES = [
  { id: 'fur', zh: '按毛色', en: 'From fur' },
  { id: 'plain', zh: '单色', en: 'Plain' },
  { id: 'off', zh: '不显示', en: 'None' },
];
export const DEFAULT_BODY = 'fur';
export const BODY_SWATCHES = [
  { hex: '#8e8b90', zh: '灰色', en: 'Grey' },
  { hex: '#c9a79a', zh: '肤色', en: 'Skin' },
];

// interaction tool on the left (the author, 11-25): stroking the fur, with its own settings. The hair dryer of v18 was
// dropped by the author the same day (it looked no different from stroking).
// Slider values are kept per tool; strength 1 = TOOL_GAIN below.
export const TOOLS = [
  { id: 'stroke', icon: 'assets/tool_stroke.png', key: 'h', zh: '抚摸', en: 'Stroke',
    hint: { zh: '在毛上按住拖动来抚摸；拖动背景旋转视角', en: 'Press and drag over the coat to stroke it; drag the background to orbit' },
    sliders: [
      { key: 'size', zh: '手掌大小', en: 'Hand size', unit: ' cm', min: 3, max: 14, step: 0.5, value: 7 },
      { key: 'force', zh: '力度', en: 'Strength', unit: '', min: 0.1, max: 2, step: 0.05, value: 1 },
      { key: 'press', zh: '下压', en: 'Press down', unit: '', min: 0, max: 1, step: 0.05, value: 0.4 },
    ] },
];
// stroke: acceleration (m/s^2) at strength 1; each guide's equilibrium moves by a / omega^2, saturated at 0.75
// of its length (shaders.js DRIVE). strokeSpeed: hand speed (m/s) at which the hand combs fully (slower = more pressing).
// settle: seconds of dynamics after the last touch while the motion is paused. decay: how fast a touched guide gets its
// rest-height floor back (s).
export const TOOL_GAIN = { stroke: 70, strokeSpeed: 0.3, settle: 5, decay: 0.35 };

export const TEXT = {
  zh: {
    title: '毛发播放器',
    subtitle: '从图像重建的动物毛发 · 实时动力学',
    statsTitle: '统计', stStrands: '毛发', stGauss: '沿毛高斯', stKernels: '细层核', stDrawn: '本帧绘制', stFps: '帧率',
    uStrands: '根', uGauss: '个', uKernels: '个', uDrawnSuffix: '个高斯',
    loading: '加载中', preparing: '准备显卡…',
    play: '播放', pause: '暂停', slow: '慢放', dynamics: '毛发动力学', compare: '对比',
    advanced: '高级', about: '关于', close: '关闭', fullscreen: '全屏', lang: 'EN',
    size: '高斯大小', density: '显示比例', colourBy: '颜色', styleBy: '画法',
    legendStrand: '每种颜色是一根毛；每个椭圆是一个沿毛高斯（2σ）。放大看更清楚。',
    legendKernels: '细层核：从锚点到当前尖端。颜色 = 尖端离开静止位置的距离 / 核的长度',
    legendRest: '静止', legendFull: '≥ 1',
    secDyn: '毛发动力学', preset: '预设', collide: '身体碰撞', floor: '不低于静止高度', reset: '从静止重新开始',
    secRender: '画面', under: '身体颜色（毛下面）', pickColour: '自选颜色', sh: '随视角变化的颜色', quality: '画质', auto: '自动', fps: '帧率',
    secGauss: '高斯视图', solo: '只看这根毛', unselect: '取消选中',
    follow: '镜头跟随', qHigh: '高', qMed: '中', qLow: '低',
    repair: '修补没拍到的部分（背中线、脚底）',
    skin: '皮肤松弛（抖动时皮毛跟着甩）',
    legendNative: '每个椭圆是一个沿毛高斯（2σ），用它渲染时的颜色',
    legendDir: '颜色 = 0.5(t+1)，t 为高斯沿毛的方向',
    legendBend: '颜色 = 这根毛当前的弯曲角（蓝 0°，红 ≥ 34°）',
    card: (c) => `这根毛由 ${c.n} 个沿毛高斯组成，长 ${c.len} mm。\n选中的是第 ${c.i} 个：长 ${c.gl} mm，宽 σ ${c.w} mm，不透明度 ${c.op}。\n它跟随 ${c.k} 个细层核运动；当前弯曲 ${c.bend}°。`,
    abCite: '引用', abLicence: '许可', abControls: '操作',
    noWebGPU: '这个浏览器不支持 WebGPU。请用 Chrome 或 Edge（113 以上）打开。',
    tooBig: (need, have) => `显卡缓冲区上限不够：需要 ${need} MB，这块显卡只给 ${have} MB。`,
    deviceLost: '显卡连接中断，请刷新页面。',
    interact: '互动', settings: '设置', toolOff: '再点一次退出',
    strandMotion: '逐根运动', strandErr: '逐根动力学无法在这块显卡上编译，已用场的核', hideStats: '隐藏（底栏的“统计”可以再打开）',
  },
  en: {
    title: 'Fur player',
    subtitle: 'Animal fur reconstructed from images · live dynamics',
    statsTitle: 'Statistics', stStrands: 'Strands', stGauss: 'Strand-aligned Gaussians', stKernels: 'Fine kernels', stDrawn: 'Drawn this frame', stFps: 'Frame rate',
    uStrands: '', uGauss: '', uKernels: '', uDrawnSuffix: 'Gaussians',
    loading: 'Loading', preparing: 'Preparing the GPU…',
    play: 'Play', pause: 'Pause', slow: 'Slow', dynamics: 'Fur dynamics', compare: 'Compare',
    advanced: 'Advanced', about: 'About', close: 'Close', fullscreen: 'Full screen', lang: '中文',
    size: 'Gaussian size', density: 'Shown', colourBy: 'Colour', styleBy: 'Draw as',
    legendStrand: 'One colour per strand; each ellipse is one strand-aligned Gaussian (2σ). Zoom in for detail.',
    legendKernels: 'Fine kernels: anchor to current tip. Colour = tip offset from rest / kernel length',
    legendRest: 'rest', legendFull: '≥ 1',
    secDyn: 'Fur dynamics', preset: 'Preset', collide: 'Body collision', floor: 'Not below rest height', reset: 'Restart from rest',
    secRender: 'Picture', under: 'Body colour (under the fur)', pickColour: 'Pick a colour', sh: 'View-dependent colour', quality: 'Quality', auto: 'Auto', fps: 'Frame rate',
    secGauss: 'Gaussian view', solo: 'Only this strand', unselect: 'Clear selection',
    follow: 'Camera follows', qHigh: 'High', qMed: 'Medium', qLow: 'Low',
    repair: 'Repair unseen fur (back midline, paw soles)',
    skin: 'Loose skin (the coat swings in a shake)',
    legendNative: 'Each ellipse is one strand-aligned Gaussian (2σ), in its rendered colour',
    legendDir: 'Colour = 0.5(t+1), t = the direction of the Gaussian along its strand',
    legendBend: 'Colour = current bend of the strand (blue 0°, red ≥ 34°)',
    card: (c) => `This strand is ${c.n} strand-aligned Gaussians, ${c.len} mm long.\nSelected: number ${c.i}, ${c.gl} mm long, width σ ${c.w} mm, opacity ${c.op}.\nIt moves with ${c.k} fine kernels; current bend ${c.bend}°.`,
    abCite: 'Citation', abLicence: 'Licences', abControls: 'Controls',
    noWebGPU: 'This browser has no WebGPU. Please open the page in Chrome or Edge (113 or later).',
    tooBig: (need, have) => `The GPU buffer limit is too small: ${need} MB needed, ${have} MB available.`,
    deviceLost: 'The GPU connection was lost; please reload the page.',
    interact: 'Interact', settings: 'Settings', toolOff: 'Click again to leave',
    strandMotion: 'Strand motion', strandErr: 'Per-strand dynamics did not compile on this GPU; using kernels', hideStats: 'Hide (Statistics in the bar shows it again)',
  },
};
export const DEFAULT_LANG = 'en';
