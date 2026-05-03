/**
 * ============================================================
 * study_teaching_engine.html
 * ------------------------------------------------------------
 * 功能：AI助学动态图片+光标标记讲解系统
 * 依赖环境：现代浏览器 (Chrome/Edge/Firefox)
 * 开发思路：
 *   1. 融合Gemini热点标记 + DeepSeek步骤教学 + 豆包交互系统的综合方案
 *   2. 左侧：动态图片展示区 + SVG光标标记 + 高亮圈
 *   3. 右侧：四段式教学面板 + 步骤导航 + 闯关检测
 *   4. 遵循AI助学全面需求.md中「去AI味、人性化、0基础友好」规范
 * 适配模块：四段式学习闭环 + 多邻国式闯关检测 + 动态图片讲解
 * 新手学习要点：HTML DOM操作 / CSS动画 / JS事件驱动 / SVG绘图 / JSON配置
 * ============================================================
 */

// =================================================================
// 第一部分：全局配置区 - 教学图片库与讲解步骤数据
// 修改这里的配置即可定制教学内容，无需改动核心逻辑代码
// =================================================================

// ------------------------------------------------------------
// 1.1 简化版问题分析配置
// ------------------------------------------------------------
var QUESTION_ANALYSIS_CONFIG = {
  subjects: {
    "物理": ["力", "运动", "能量", "牛顿", "加速度", "质量", "速度", "位移"],
    "数学": ["函数", "方程", "不等式", "几何", "三角", "导数", "概率", "数列"],
    "化学": ["反应", "元素", "化合物", "摩尔", "化学式", "方程", "氧化"]
  },
  
  questionTypes: {
    "基础概念": ["是什么", "定义", "概念", "含义", "什么叫"],
    "原理分析": ["为什么", "原理", "原因", "解释", "推导"],
    "计算应用": ["计算", "求", "等于多少", "多少"],
    "综合分析": ["分析", "讨论", "比较", "区别", "联系"]
  },
  
  teachingStages: {
    "基础概念": "parse",
    "原理分析": "principle", 
    "计算应用": "test",
    "综合分析": "summary"
  }
};

// ------------------------------------------------------------
// 1.2 知识点库
// ------------------------------------------------------------
var KNOWLEDGE_POINTS_LIBRARY = [
  { subject: "物理", module: "牛顿第二定律", keywords: ["牛顿第二定律", "F=ma", "加速度", "合外力"], imageIndex: 0 },
  { subject: "物理", module: "受力分析", keywords: ["受力分析", "重力", "弹力", "摩擦力"], imageIndex: 1 },
  { subject: "物理", module: "力学综合", keywords: ["力学", "运动学", "动力学"], imageIndex: 2 }
];

// ------------------------------------------------------------
// 1.3 简化版问题分析器
// ------------------------------------------------------------
var QuestionAnalyzer = function() {
  var self = this;
  
  self.analyze = function(questionText) {
    var result = {
      subject: "物理",
      module: "牛顿第二定律",
      questionType: "计算应用",
      teachingStage: "parse",
      confidence: 0.8,
      suggestion: ""
    };
    
    if (!questionText || questionText.trim() === "") {
      result.suggestion = "请输入您想要学习的问题";
      return result;
    }
    
    var text = questionText.trim();
    
    // 识别学科
    for (var subject in QUESTION_ANALYSIS_CONFIG.subjects) {
      if (QUESTION_ANALYSIS_CONFIG.subjects[subject].some(function(k) { return text.indexOf(k) !== -1; })) {
        result.subject = subject;
        break;
      }
    }
    
    // 识别题型
    for (var type in QUESTION_ANALYSIS_CONFIG.questionTypes) {
      if (QUESTION_ANALYSIS_CONFIG.questionTypes[type].some(function(k) { return text.indexOf(k) !== -1; })) {
        result.questionType = type;
        result.teachingStage = QUESTION_ANALYSIS_CONFIG.teachingStages[type];
        break;
      }
    }
    
    // 匹配知识点
    for (var i = 0; i < KNOWLEDGE_POINTS_LIBRARY.length; i++) {
      var kp = KNOWLEDGE_POINTS_LIBRARY[i];
      if (kp.subject === result.subject) {
        if (kp.keywords.some(function(k) { return text.indexOf(k) !== -1; })) {
          result.module = kp.module;
          result.imageIndex = kp.imageIndex;
          result.confidence = 0.9;
          break;
        }
      }
    }
    
    result.suggestion = "已识别：【" + result.subject + "】·【" + result.module + "】，" + result.questionType + "题型";
    
    return result;
  };
};

// 创建全局问题分析器实例
var questionAnalyzer = new QuestionAnalyzer();

/**
 * TEACHING_IMAGE_LIBRARY - 教学图片库配置
 * 每项包含图片路径、描述、预设的光标标记坐标
 * 
 * 修改方式：替换 path 为你的图片路径，调整 marks 坐标数组
 * 坐标说明：x和y使用百分比(0-100)，相对于图片左上角
 * 
 * 报错留意：
 *   - 图片不显示：检查 path 路径是否正确，图片是否与HTML同目录
 *   - 标记位置偏移：调整 marks 中的 x/y 百分比值
 */
var TEACHING_IMAGE_LIBRARY = [
  {
    id: "img_newton_001",
    path: "https://picsum.photos/seed/physics1/800/500",
    title: "牛顿第二定律演示",
    subject: "物理",
    module: "牛顿第二定律",
    description: "展示牛顿第二定律 F=ma 的基本原理",
    marks: [
      { x: 30, y: 30, label: "力 F (单位：N)" },
      { x: 50, y: 50, label: "质量 m (单位：kg)" },
      { x: 70, y: 30, label: "加速度 a (单位：m/s²)" }
    ]
  },
  {
    id: "img_newton_002",
    path: "https://picsum.photos/seed/physics2/800/500",
    title: "受力分析示意图",
    subject: "物理",
    module: "牛顿第二定律·受力分析",
    description: "展示物体在力作用下的运动分析",
    marks: [
      { x: 25, y: 40, label: "推力 F" },
      { x: 50, y: 70, label: "摩擦力 f" },
      { x: 75, y: 40, label: "合外力 F合" }
    ]
  },
  {
    id: "img_newton_003",
    path: "https://picsum.photos/seed/physics3/800/500",
    title: "F=ma关系图",
    subject: "物理",
    module: "牛顿第二定律·公式理解",
    description: "展示力、质量、加速度三者的定量关系",
    marks: [
      { x: 50, y: 50, label: "F = m × a" },
      { x: 20, y: 70, label: "m = F/a" },
      { x: 80, y: 70, label: "a = F/m" }
    ]
  },
  {
    id: "img_newton_004",
    path: "https://picsum.photos/seed/physics4/800/500",
    title: "电梯问题分析",
    subject: "物理",
    module: "牛顿第二定律·超重失重",
    description: "展示电梯中物体的超重与失重现象",
    marks: [
      { x: 50, y: 25, label: "支持力 N" },
      { x: 50, y: 55, label: "重力 G = mg" },
      { x: 50, y: 80, label: "合外力 F = N - G" }
    ]
  },
  {
    id: "img_newton_005",
    path: "https://picsum.photos/seed/physics5/800/500",
    title: "斜面受力分析",
    subject: "物理",
    module: "牛顿第二定律·斜面问题",
    description: "展示斜面上物体的受力与运动分析",
    marks: [
      { x: 50, y: 25, label: "重力 mg" },
      { x: 30, y: 50, label: "支持力 N" },
      { x: 70, y: 50, label: "下滑力 mgsinθ" }
    ]
  },
  {
    id: "img_newton_006",
    path: "https://picsum.photos/seed/physics6/800/500",
    title: "连接体问题",
    subject: "物理",
    module: "牛顿第二定律·整体法",
    description: "展示多个物体连接在一起的运动分析",
    marks: [
      { x: 30, y: 40, label: "m₁ 受力分析" },
      { x: 50, y: 60, label: "细绳张力 T" },
      { x: 70, y: 40, label: "m₂ 受力分析" }
    ]
  }
];

/**
 * TEACHING_STEPS - 四段式教学步骤配置
 * 
 * 结构说明（对应AI助学需求第二条「核心强制工作流程」）：
 *   stage: "parse"       → 第一阶段：解析问题
 *   stage: "principle"   → 第二阶段：讲解底层原理
 *   stage: "test"        → 第三阶段：变式测试检验
 *   stage: "summary"     → 第四阶段：归纳归档总结
 * 
 * 每阶段含多个steps，每个step定义：
 *   title       - 当前步骤标题
 *   content     - 讲解内容文本（支持 \n 换行）
 *   cursorX/Y   - 光标在图片上的百分比位置
 *   ring        - 高亮圈配置 {x, y, size} 或 null
 *   isQuiz      - 是否为闯关选择题（仅在test阶段）
 *   quizOptions - 选择题选项数组
 *   quizAnswer  - 正确答案索引（从0开始）
 * 
 * 修改方式：增删 steps 数组内容，调整坐标/文字即可
 */
var TEACHING_STEPS = [
  {
    stage: "parse",
    stageTitle: "一、解析问题",
    icon: "search",
    steps: [
      {
        title: "题目展示",
        content: "【例题】一个质量为 5kg 的物体，受到 20N 的水平拉力作用，求物体的加速度大小。",
        cursorX: 50,
        cursorY: 50,
        ring: null
      },
      {
        title: "画受力图",
        content: "首先，对物体进行受力分析：\n\n① 重力 G = mg = 5kg × 10m/s² = 50N（竖直向下）\n② 支持力 N（竖直向上，与重力平衡）\n③ 拉力 F = 20N（水平向右）\n④ 摩擦力 f = 0（无摩擦）\n\n水平方向：只有拉力 F → F合 = 20N",
        cursorX: 30,
        cursorY: 40,
        ring: { x: 30, y: 40, size: 60 }
      },
      {
        title: "找已知量",
        content: "从题目中提取已知条件：\n\n  ● 质量 m = 5 kg\n  ● 拉力 F = 20 N\n  ● 摩擦力 f = 0 N\n  ● 方向：水平\n\n求解目标：加速度 a\n\n注意：单位必须统一使用国际单位制！",
        cursorX: 70,
        cursorY: 30,
        ring: { x: 70, y: 30, size: 55 }
      }
    ]
  },
  {
    stage: "principle",
    stageTitle: "二、讲解原理",
    icon: "lightbulb",
    steps: [
      {
        title: "定律内容",
        content: "牛顿第二定律（Newton's Second Law）：\n\n物体的加速度与所受合外力成正比，与物体质量成反比。\n\n【公式】F = ma\n\n各物理量含义：\n  ● F —— 合外力，单位 N（牛顿）\n  ● m —— 质量，单位 kg（千克）\n  ● a —— 加速度，单位 m/s²（米每二次方秒）",
        cursorX: 50,
        cursorY: 30,
        ring: { x: 50, y: 30, size: 65 }
      },
      {
        title: "公式推导",
        content: "由 F = ma 可推导求 a：\n\n    a = F/m\n      = 20N / 5kg\n      = 20(kg·m/s²) / 5kg\n      = 4 m/s²\n\n【方向判断】\n加速度方向与合外力方向相同\n\n因为 F合 = 20N 方向向右\n所以 a = 4m/s² 方向向右 →",
        cursorX: 50,
        cursorY: 55,
        ring: { x: 50, y: 55, size: 58 }
      },
      {
        title: "规范解题",
        content: "【牛顿第二定律规范解题步骤】\n\n① 选对象 —— 确定研究物体\n② 看受力 —— 画受力分析图\n③ 求合力 —— 水平/竖直方向分别求\n④ 列方程 —— F合 = ma\n⑤ 代数据 —— 统一单位后再代入\n⑥ 检验 —— 检查结果是否符合实际",
        cursorX: 35,
        cursorY: 45,
        ring: { x: 35, y: 45, size: 52 }
      }
    ]
  },
  {
    stage: "test",
    stageTitle: "三、变式测试",
    icon: "quiz",
    steps: [
      {
        title: "变式一",
        content: "【变式练习1】\n\n质量为 2kg 的物体，受到 6N 的水平拉力。若同时受到 2N 的摩擦力（方向与拉力相反），求物体的加速度。\n\n【分析】\n  ● F合 = F拉 - f摩擦 = 6N - 2N = 4N\n  ● 方向：与拉力方向相同（向右）",
        cursorX: 50,
        cursorY: 40,
        ring: null,
        isQuiz: true,
        quizOptions: ["A. 1 m/s²", "B. 2 m/s²", "C. 3 m/s²", "D. 4 m/s²"],
        quizAnswer: 1,
        quizExplain: "【解答】\nF合 = 6N - 2N = 4N\na = F合/m = 4N ÷ 2kg = 2 m/s²\n\n答案：B\n\n【易错提醒】\n摩擦力方向与运动趋势相反，不是与拉力相反！"
      },
      {
        title: "变式二",
        content: "【变式练习2】\n\n质量为 3kg 的物体，受到三个水平力作用：\nF₁ = 8N 向右，F₂ = 5N 向左，F₃ = 2N 向右。\n求物体的加速度。\n\n【分析】\n  ● 设向右为正方向\n  ● F合 = F₁ - F₂ + F₃ = 8 - 5 + 2 = 5N\n  ● 方向：向右",
        cursorX: 50,
        cursorY: 40,
        ring: null,
        isQuiz: true,
        quizOptions: ["A. 5/3 m/s² 向右", "B. 5/3 m/s² 向左", "C. 5 m/s² 向右", "D. 5 m/s² 向左"],
        quizAnswer: 0,
        quizExplain: "【解答】\nF合 = 8 - 5 + 2 = 5N（向右）\na = F合/m = 5N ÷ 3kg = 5/3 m/s²\n方向：向右\n\n答案：A\n\n【技巧】先规定正方向，再带符号运算！"
      }
    ]
  },
  {
    stage: "summary",
    stageTitle: "四、归纳总结",
    icon: "notebook",
    steps: [
      {
        title: "知识要点",
        content: "【牛顿第二定律核心要点】\n\n① 基本公式：F = ma\n② 方向一致：a 的方向与 F合 的方向相同\n③ 瞬时性：F 变化，a 同时变化（同时开始、同时停止）\n④ 独立性：每个力单独产生自己的加速度\n\n【三种常见题型】\n  ● 已知 F、m → 求 a（直接代入公式）\n  ● 已知 a、m → 求 F（F = ma）\n  ● 已知 F、a → 求 m（m = F/a）",
        cursorX: 30,
        cursorY: 40,
        ring: { x: 30, y: 40, size: 58 }
      },
      {
        title: "易错警示",
        content: "【考试常见错误TOP5】\n\n① 单位未统一\n  ✗ F=20N, m=5kg → a=Fm=4\n  ✓ 必须写成 a = 4 m/s²\n\n② 方向判断错误\n  ✗ 认为加速度方向与拉力相反\n  ✓ a 方向与 F合 方向相同\n\n③ 漏掉某个力\n  ✓ 一定要画完整的受力图！\n\n④ 质量与重力混淆\n  ✗ 把 g=10m/s² 当成质量\n  ✓ G=mg 是重力，不是质量\n\n⑤ 多物体忘记隔离\n  ✗ 连接体问题直接相加\n  ✓ 先隔离分析，再整体验证",
        cursorX: 70,
        cursorY: 45,
        ring: { x: 70, y: 45, size: 55 }
      },
      {
        title: "记忆口诀",
        content: "【牛顿第二定律记忆口诀】\n\n合外力定加速度，\n质量惯性是抵抗。\n方向永远同一致，\nF等m乘a记心上。\n\n【口诀释义】\n  合外力 → 决定加速度的大小和方向\n  质量   → 质量越大，惯性越大，加速度越小\n  同方向 → 加速度方向总与合外力方向一致\n  F=ma  → 三个物理量的因果关系",
        cursorX: 50,
        cursorY: 50,
        ring: { x: 50, y: 50, size: 52 }
      }
    ]
  }
];

// =================================================================
// 第二部分：核心引擎 - 动态图片加载与光标标记系统
// 以下为核心逻辑代码，新手可对照注释逐步理解
// =================================================================

/**
 * TeachingEngine - 教学引擎主控制器
 * 
 * 作用：管理整个教学演示的生命周期
 * 包括：图片加载、光标移动、步骤切换、闯关检测、状态管理
 * 
 * 使用方式：
 *   var engine = new TeachingEngine();
 *   engine.init(); // 初始化并启动
 * 
 * 修改方式：
 *   配置在最上方的 TEACHING_IMAGE_LIBRARY 和 TEACHING_STEPS 中修改
 *   核心逻辑一般无需改动
 */
var TeachingEngine = function() {
  // 内部状态变量（新手注意：var定义的变量仅在函数作用域内有效）
  var self = this;
  
  // 当前选中的图片索引（从0开始）
  self.currentImageIndex = 0;
  
  // 当前四段式教学阶段索引
  self.currentStageIndex = 0;
  
  // 当前阶段内步骤索引
  self.currentStepIndex = 0;
  
  // 自动播放状态
  self.isAutoPlaying = false;
  self.autoPlayTimer = null;
  
  // 闯关结果统计
  self.quizResults = { total: 0, correct: 0, wrong: 0 };
  
  // DOM元素引用缓存（避免反复查询，提升性能）
    self.dom = {
      imagePanel: null,     // 图片展示容器
      teachingImage: null,  // 教学图片元素
      cursorMarker: null,   // 光标标记SVG元素
      ringContainer: null,  // 高亮圈容器
      stageTitle: null,     // 阶段标题
      stepTitle: null,      // 步骤标题
      stepContent: null,     // 步骤内容
      quizZone: null,       // 闯关题目区域
      quizOptions: null,    // 选择题选项容器
      quizFeedback: null,    // 闯关反馈区
      stepIndicator: null,  // 步骤指示器
      progressBar: null,     // 进度条
      imageThumbnails: null,// 图片缩略图列表
      playBtn: null,        // 播放按钮
      prevBtn: null,        // 上一步按钮
      nextBtn: null,        // 下一步按钮
      autoGuideBtn: null,  // 自动引导按钮
      stageDots: [],        // 阶段指示点
      quizStats: null       // 闯关统计
    };
  
  /**
   * init - 初始化教学引擎
   * 
   * 作用：绑定DOM元素、渲染图片缩略图、绑定事件、进入首页
   * 调用时机：页面加载完成后自动调用
   * 
   * 底层原理：
   *   1. 获取所有需要的DOM元素引用并缓存
   *   2. 根据图片库生成缩略图列表
   *   3. 为按钮绑定点击/键盘事件
   *   4. 加载第一张图片并显示第一步讲解
   */
  self.init = function() {
    // 获取DOM元素引用
    self.dom.imagePanel = document.getElementById('imagePanel');
    self.dom.teachingImage = document.getElementById('teachingImage');
    self.dom.cursorMarker = document.getElementById('cursorMarker');
    self.dom.ringContainer = document.getElementById('ringContainer');
    self.dom.stageTitle = document.getElementById('stageTitle');
    self.dom.stepTitle = document.getElementById('stepTitle');
    self.dom.stepContent = document.getElementById('stepContent');
    self.dom.quizZone = document.getElementById('quizZone');
    self.dom.quizOptions = document.getElementById('quizOptions');
    self.dom.quizFeedback = document.getElementById('quizFeedback');
    self.dom.stepIndicator = document.getElementById('stepIndicator');
    self.dom.progressBar = document.getElementById('progressBar');
    self.dom.imageThumbnails = document.getElementById('imageThumbnails');
    self.dom.playBtn = document.getElementById('playBtn');
    self.dom.prevBtn = document.getElementById('prevBtn');
    self.dom.nextBtn = document.getElementById('nextBtn');
    self.dom.autoGuideBtn = document.getElementById('autoGuideBtn');
    self.dom.quizStats = document.getElementById('quizStats');
    self.dom.answerReveal = document.getElementById('answerReveal');
    self.dom.answerExplain = document.getElementById('answerExplain');
    
    // 获取所有阶段指示点
    self.dom.stageDots = document.querySelectorAll('.stage-dot');
    
    // 渲染图片缩略图列表
    self.renderThumbnails();
    
    // 绑定按钮事件
    self.bindEvents();
    
    // 加载第一张图片，初始化显示
    self.loadImage(0);
  };
  
  /**
   * configureForQuestion - 根据用户问题自动配置教学内容
   * 
   * 作用：分析用户输入的问题，自动匹配知识点并配置教学内容
   * 
   * 参数：
   *   questionText - 用户输入的问题文本
   * 
   * 返回值：分析结果对象，包含学科、知识点、题型、难度等信息
   */
  self.configureForQuestion = function(questionText) {
    // 调用问题分析器分析问题
    var analysis = questionAnalyzer.analyze(questionText);
    
    // 显示分析结果
    self.showAnalysisResult(analysis);
    
    // 如果识别到知识点，自动跳转到对应的教学阶段
    if (analysis.confidence >= 0.5) {
      // 跳转到推荐的教学阶段
      var stageMap = {
        "parse": 0,
        "principle": 1,
        "test": 2,
        "summary": 3
      };
      
      var targetStage = stageMap[analysis.teachingStage] || 0;
      
      // 如果有匹配的图片索引，跳转到相关图片
      if (analysis.imageIndex !== undefined && analysis.imageIndex >= 0) {
        self.loadImage(analysis.imageIndex);
      }
      
      // 跳转到对应阶段的第一步
      self.currentStageIndex = targetStage;
      self.currentStepIndex = 0;
      self.updateDisplay();
      
      // 根据分析结果设置科目标签
      if (analysis.subject) {
        var badge = document.getElementById('subjectBadge');
        if (badge) {
          badge.textContent = analysis.subject + ' | ' + (analysis.module || '通用知识');
        }
      }
    }
    
    return analysis;
  };
  
  /**
   * showAnalysisResult - 显示问题分析结果
   */
  self.showAnalysisResult = function(analysis) {
    var resultContainer = document.getElementById('analysisResult');
    if (!resultContainer) {
      return;
    }
    
    // 构建分析结果HTML
    var html = '<div class="analysis-header">问题分析结果</div>';
    html += '<div class="analysis-item"><span class="label">学科：</span>' + analysis.subject + '</div>';
    html += '<div class="analysis-item"><span class="label">模块：</span>' + (analysis.module || '未识别') + '</div>';
    html += '<div class="analysis-item"><span class="label">题型：</span>' + analysis.questionType + '</div>';
    html += '<div class="analysis-item"><span class="label">匹配度：</span>' + Math.round(analysis.confidence * 100) + '%</div>';
    html += '<div class="analysis-suggestion">' + analysis.suggestion + '</div>';
    
    resultContainer.innerHTML = html;
    resultContainer.style.display = 'block';
    
    // 5秒后自动隐藏
    setTimeout(function() {
      resultContainer.style.display = 'none';
    }, 5000);
  };
  
  /**
   * renderThumbnails - 渲染媒体缩略图列表
   * 
   * 作用：根据 TEACHING_IMAGE_LIBRARY 动态生成左侧缩略图
   * 支持图片和视频两种类型
   * 
   * 底层原理：
   *   - 遍历媒体库数组
   *   - 为每个媒体创建缩略图DOM节点
   *   - 为每个缩略图绑定点击切换事件
   */
  self.renderThumbnails = function() {
    var container = self.dom.imageThumbnails;
    container.innerHTML = ''; // 清空已有内容
    
    for (var i = 0; i < TEACHING_IMAGE_LIBRARY.length; i++) {
      (function(index) {
        // 自执行函数创建闭包，保存每次循环的index值
        // 为什么这么写：如果不用闭包，所有点击事件都会使用循环结束后的i值
        var imgData = TEACHING_IMAGE_LIBRARY[index];
        
        var thumb = document.createElement('div');
        thumb.className = 'thumbnail-item';
        if (index === self.currentImageIndex) {
          thumb.classList.add('active');
        }
        
        // 缩略图图片
        var img = document.createElement('img');
        img.src = imgData.path;
        img.alt = imgData.title;
        img.loading = 'lazy'; // 懒加载优化
        
        // 缩略图标题
        var label = document.createElement('span');
        label.className = 'thumbnail-label';
        label.textContent = imgData.subject + ' | ' + imgData.module;
        
        thumb.appendChild(img);
        thumb.appendChild(label);
        
        // 点击缩略图切换图片
        thumb.addEventListener('click', function() {
          self.loadImage(index);
        });
        
        container.appendChild(thumb);
      })(i);
    }
  };
  
  /**
   * loadImage - 加载并切换教学图片
   * 
   * 作用：切换左侧展示的图片，同时更新缩略图高亮状态
   * 
   * 参数：
   *   index - 图片在 TEACHING_IMAGE_LIBRARY 中的索引
   * 
   * 底层原理：
   *   - 修改img元素的src属性触发浏览器加载新图片
   *   - onload事件确保图片加载完成后再计算光标位置
   *   - 重置阶段和步骤回到教学内容的开始
   */
  self.loadImage = function(index) {
    if (index < 0 || index >= TEACHING_IMAGE_LIBRARY.length) return;
    
    self.currentImageIndex = index;
    
    var imgData = TEACHING_IMAGE_LIBRARY[index];
    var img = self.dom.teachingImage;
    
    // 修改图片源地址
    img.src = imgData.path;
    img.alt = imgData.title;
    
    // 更新缩略图激活状态
    var thumbs = self.dom.imageThumbnails.querySelectorAll('.thumbnail-item');
    for (var i = 0; i < thumbs.length; i++) {
      thumbs[i].classList.remove('active');
    }
    if (thumbs[index]) {
      thumbs[index].classList.add('active');
    }
    
    // 立即更新显示（不等待图片加载）
    // 这样可以确保用户立即看到内容
    self.currentStageIndex = 0;
    self.currentStepIndex = 0;
    self.updateDisplay();
  };
  
  /**
   * getCurrentStep - 获取当前教学步骤的配置数据
   * 
   * 返回值：包含当前步骤所有配置的对象
   */
  self.getCurrentStep = function() {
    var stage = TEACHING_STEPS[self.currentStageIndex];
    if (!stage) return null;
    return stage.steps[self.currentStepIndex] || null;
  };
  
  /**
   * updateDisplay - 更新整个界面的显示状态
   * 
   * 作用：核心刷新函数，每次切换步骤时调用
   * 更新内容包括：阶段标题、步骤标题、讲解内容、光标位置、
   *   高亮圈、闯关题目、进度条、阶段指示点、按钮状态
   * 
   * 调用时机：loadImage后、点击上一步/下一步时、自动播放切换时
   */
  self.updateDisplay = function() {
    var stage = TEACHING_STEPS[self.currentStageIndex];
    var step = self.getCurrentStep();
    if (!stage || !step) return;
    
    // ---- 1. 更新右侧文字内容 ----
    self.dom.stageTitle.textContent = stage.stageTitle;
    self.dom.stepTitle.textContent = step.title;
    self.dom.stepContent.textContent = step.content;
    
    // ---- 2. 更新阶段指示点 ----
    for (var i = 0; i < self.dom.stageDots.length; i++) {
      var dot = self.dom.stageDots[i];
      dot.classList.remove('active', 'completed');
      if (i === self.currentStageIndex) {
        dot.classList.add('active');
      } else if (i < self.currentStageIndex) {
        dot.classList.add('completed');
      }
    }
    
    // ---- 3. 更新光标位置（百分比坐标转像素坐标） ----
    self.updateCursorPosition(step);
    
    // ---- 4. 更新高亮标记圈 ----
    self.updateHighlightRing(step);
    
    // ---- 5. 处理闯关题目 ----
    self.updateQuizSection(step);
    
    // ---- 6. 更新进度条 ----
    self.updateProgress();
    
    // ---- 7. 更新步骤指示文字 ----
    self.updateStepIndicator();
    
    // ---- 8. 更新按钮状态 ----
    self.updateButtonStates();
    
    // ---- 9. 更新闯关统计 ----
    self.updateQuizStats();
  };
  
  /**
   * updateCursorPosition - 计算并移动光标到目标位置
   * 
   * 参数：
   *   step - 当前步骤的配置对象，包含 cursorX, cursorY
   * 
   * 底层原理：
   *   1. 获取图片在页面中的实际显示尺寸（getBoundingClientRect）
   *   2. 用百分比 × 实际宽高 = 像素坐标
   *   3. 用CSS left/top + transition 实现平滑移动动画
   *   4. 到达后触发缩放动画（scale放大再还原，模拟点击效果）
   */
  self.updateCursorPosition = function(step) {
    var cursor = self.dom.cursorMarker;
    var img = self.dom.teachingImage;
    var panel = self.dom.imagePanel;
    
    if (!step || !img || !panel || !cursor) return;
    
    // 如果图片还没有尺寸，先使用百分比定位
    cursor.style.left = step.cursorX + '%';
    cursor.style.top = step.cursorY + '%';
    
    // 到达后触发缩放动画（模拟点击/指向效果）
    cursor.style.transform = 'translate(-50%, -50%) scale(1.25)';
    setTimeout(function() {
      cursor.style.transform = 'translate(-50%, -50%) scale(1)';
    }, 300);
  };
  
  /**
   * updateHighlightRing - 更新高亮标记圈
   * 
   * 作用：在图片指定位置显示金色闪烁高亮圈
   * 
   * 参数：
   *   step - 当前步骤配置
   * 
   * 底层原理：
   *   动态创建div元素，用border-radius:50%画圆
   *   用CSS animation实现脉冲闪烁效果
   *   先清除旧圈，再添加新圈（避免重复叠加）
   */
  self.updateHighlightRing = function(step) {
    var container = self.dom.ringContainer;
    container.innerHTML = ''; // 清除旧圈
    
    if (!step.ring) return; // 如果当前步骤不需要高亮圈，直接返回
    
    // 创建高亮圈DOM元素
    var ring = document.createElement('div');
    ring.className = 'highlight-ring';
    
    // 设置圈的大小
    var size = step.ring.size;
    ring.style.width = size + 'px';
    ring.style.height = size + 'px';
    
    // 使用百分比定位（更简单可靠）
    ring.style.left = step.ring.x + '%';
    ring.style.top = step.ring.y + '%';
    ring.style.transform = 'translate(-50%, -50%)';
    
    container.appendChild(ring);
    
    // 添加动画类
    setTimeout(function() {
      ring.classList.add('active');
    }, 10);
  };
  
  /**
   * updateQuizSection - 处理闯关题目区域
   * 
   * 作用：当步骤标记为isQuiz时，显示选择题UI
   * 
   * 参数：
   *   step - 当前步骤配置
   * 
   * 底层原理：
   *   - 检查step.isQuiz字段判断是否为闯关步骤
   *   - 动态生成选项按钮
   *   - 绑定点击事件进行对错判定
   *   - 即时反馈（正确→绿色提示，错误→红色提示+显示正确答案）
   */
  self.updateQuizSection = function(step) {
    var quizZone = self.dom.quizZone;
    var quizOptions = self.dom.quizOptions;
    var quizFeedback = self.dom.quizFeedback;
    
    if (!step.isQuiz) {
      // 非闯关步骤，隐藏闯关区域
      quizZone.style.display = 'none';
      return;
    }
    
    // 闯关步骤，显示闯关区域
    quizZone.style.display = 'block';
    quizOptions.innerHTML = '';
    quizFeedback.style.display = 'none';
    
    // 动态生成选项按钮
    for (var i = 0; i < step.quizOptions.length; i++) {
      (function(optionIndex) {
        // 闭包保存选项索引（原理同renderThumbnails中的闭包）
        var btn = document.createElement('button');
        btn.className = 'quiz-option-btn';
        btn.textContent = step.quizOptions[optionIndex];
        
        // 选项点击事件
        btn.addEventListener('click', function() {
          self.handleQuizAnswer(optionIndex, step);
        });
        
        quizOptions.appendChild(btn);
      })(i);
    }
  };
  
  /**
   * handleQuizAnswer - 处理闯关答案选择
   * 
   * 参数：
   *   selectedIndex - 用户选择的选项索引
   *   step - 当前步骤配置（含正确答案）
   * 
   * 底层原理：
   *   极简判定算法：selectedIndex === quizAnswer
   *   正确：记录正确数，绿色反馈，递进到下一步
   *   错误：记录错误数，红色反馈，显示解析，允许重试
   */
  self.handleQuizAnswer = function(selectedIndex, step) {
    var quizFeedback = self.dom.quizFeedback;
    var allBtns = self.dom.quizOptions.querySelectorAll('.quiz-option-btn');
    
    // 禁用所有选项按钮（防止重复点击）
    for (var i = 0; i < allBtns.length; i++) {
      allBtns[i].disabled = true;
    }
    
    self.quizResults.total++;
    
    if (selectedIndex === step.quizAnswer) {
      // 回答正确
      self.quizResults.correct++;
      quizFeedback.style.display = 'block';
      quizFeedback.className = 'quiz-feedback correct';
      quizFeedback.innerHTML = '<span class="feedback-icon">&#10003;</span> 回答正确！';
      
      // 标记正确选项为绿色
      allBtns[selectedIndex].classList.add('correct');
      
      // 1.5秒后自动进入下一步
      var _self = self;
      setTimeout(function() {
        _self.nextStep();
      }, 1500);
    } else {
      // 回答错误
      self.quizResults.wrong++;
      quizFeedback.style.display = 'block';
      quizFeedback.className = 'quiz-feedback wrong';
      quizFeedback.innerHTML = '<span class="feedback-icon">&#10007;</span> 再想想看，答案不是这个哦';
      
      // 标记错误选项为红色
      allBtns[selectedIndex].classList.add('wrong');
      
      // 显示答案揭示按钮
      var revealBtn = document.createElement('button');
      revealBtn.className = 'quiz-reveal-btn';
      revealBtn.textContent = '查看解析';
      revealBtn.onclick = function() {
        self.dom.answerReveal.style.display = 'block';
        self.dom.answerReveal.innerHTML = '<strong>正确答案：</strong>' + step.quizOptions[step.quizAnswer];
        self.dom.answerExplain.style.display = 'block';
        self.dom.answerExplain.innerHTML = '<strong>解析：</strong>' + step.quizExplain;
      };
      quizFeedback.appendChild(revealBtn);
      
      // 标记正确答案为绿色
      allBtns[step.quizAnswer].classList.add('correct');
      
      // 恢复按钮（允许重试）
      for (var j = 0; j < allBtns.length; j++) {
        if (j !== selectedIndex) {
          allBtns[j].disabled = false;
        }
      }
    }
    
    self.updateQuizStats();
  };
  
  /**
   * updateProgress - 更新进度条
   * 
   * 底层原理：
   *   总步骤数 = TEACHING_STEPS 所有阶段的steps数量之和
   *   已完成步骤数 = 之前阶段的步骤 + 当前阶段已完成步骤
   *   进度百分比 = 已完成 / 总步骤数 × 100
   */
  self.updateProgress = function() {
    // 统计总步骤数
    var totalSteps = 0;
    for (var i = 0; i < TEACHING_STEPS.length; i++) {
      totalSteps += TEACHING_STEPS[i].steps.length;
    }
    
    // 统计已完成步骤数
    var completedSteps = 0;
    for (var i = 0; i < self.currentStageIndex; i++) {
      completedSteps += TEACHING_STEPS[i].steps.length;
    }
    completedSteps += self.currentStepIndex;
    
    // 计算百分比
    var percentage = Math.round((completedSteps / totalSteps) * 100);
    self.dom.progressBar.style.width = percentage + '%';
  };
  
  /**
   * updateStepIndicator - 更新步骤指示文字
   */
  self.updateStepIndicator = function() {
    var stage = TEACHING_STEPS[self.currentStageIndex];
    var totalInStage = stage.steps.length;
    self.dom.stepIndicator.textContent = 
      '阶段 ' + (self.currentStageIndex + 1) + '/' + TEACHING_STEPS.length + 
      ' | 步骤 ' + (self.currentStepIndex + 1) + '/' + totalInStage;
  };
  
  /**
   * updateButtonStates - 更新导航按钮的启用/禁用状态
   * 
   * 底层原理：
   *   首步禁用「上一步」按钮
   *   末步禁用「下一步」按钮
   *   防止用户越界操作
   */
  self.updateButtonStates = function() {
    var isFirstStep = (self.currentStageIndex === 0 && self.currentStepIndex === 0);
    var isLastStep = self.isLastStep();
    
    self.dom.prevBtn.disabled = isFirstStep;
    self.dom.nextBtn.disabled = isLastStep;
  };
  
  /**
   * isLastStep - 判断当前是否为最后一步
   */
  self.isLastStep = function() {
    var lastStage = TEACHING_STEPS[TEACHING_STEPS.length - 1];
    return (
      self.currentStageIndex === TEACHING_STEPS.length - 1 &&
      self.currentStepIndex === lastStage.steps.length - 1
    );
  };
  
  /**
   * updateQuizStats - 更新闯关统计数据展示
   */
  self.updateQuizStats = function() {
    var stats = self.dom.quizStats;
    if (self.quizResults.total === 0) {
      stats.textContent = '暂无闯关记录';
    } else {
      stats.textContent = 
        '已闯关：' + self.quizResults.total + ' 题 | ' +
        '正确：' + self.quizResults.correct + ' | ' +
        '错误：' + self.quizResults.wrong;
    }
  };
  
  // =================================================================
  // 第三部分：导航控制 - 上一步/下一步/自动播放/键盘控制
  // =================================================================
  
  /**
   * nextStep - 进入下一步
   * 
   * 底层原理：
   *   1. 检查当前阶段是否还有剩余步骤
   *   2. 有则 stepIndex+1
   *   3. 无则进入下一阶段，stepIndex归零
   *   4. 到达最后一步时自动停止播放
   */
  self.nextStep = function() {
    if (self.isLastStep()) {
      self.stopAutoPlay();
      return;
    }
    
    var currentStage = TEACHING_STEPS[self.currentStageIndex];
    
    if (self.currentStepIndex < currentStage.steps.length - 1) {
      // 当前阶段还有剩余步骤
      self.currentStepIndex++;
    } else {
      // 进入下一阶段
      self.currentStageIndex++;
      self.currentStepIndex = 0;
    }
    
    self.updateDisplay();
  };
  
  /**
   * prevStep - 返回上一步
   */
  self.prevStep = function() {
    if (self.currentStageIndex === 0 && self.currentStepIndex === 0) {
      return; // 已在第一步，无法后退
    }
    
    if (self.currentStepIndex > 0) {
      self.currentStepIndex--;
    } else {
      self.currentStageIndex--;
      self.currentStepIndex = TEACHING_STEPS[self.currentStageIndex].steps.length - 1;
    }
    
    self.updateDisplay();
  };
  
  /**
   * toggleAutoPlay - 切换自动播放状态
   * 
   * 底层原理：
   *   使用 setInterval 定时器，每3秒自动调用 nextStep()
   *   到达最后一步自动停止
   *   再次点击暂停 → clearInterval 清除定时器
   */
  self.toggleAutoPlay = function() {
    if (self.isAutoPlaying) {
      self.stopAutoPlay();
    } else {
      self.startAutoPlay();
    }
  };
  
  /**
   * startAutoPlay - 开始自动播放
   */
  self.startAutoPlay = function() {
    self.isAutoPlaying = true;
    self.dom.playBtn.innerHTML = 
      '<svg viewBox="0 0 24 24" width="18" height="18"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> 暂停';
    self.dom.playBtn.classList.add('playing');
    
    // 如果已是最后一步，从头开始
    if (self.isLastStep()) {
      self.currentStageIndex = 0;
      self.currentStepIndex = 0;
      self.updateDisplay();
    }
    
    // 设置定时器，每3000ms自动前进一步
    // setInterval返回的ID用于后续clearInterval停止
    var _self = self;
    self.autoPlayTimer = setInterval(function() {
      if (_self.isLastStep()) {
        _self.stopAutoPlay();
      } else {
        _self.nextStep();
      }
    }, 3000);
  };
  
  /**
   * stopAutoPlay - 停止自动播放
   */
  self.stopAutoPlay = function() {
    self.isAutoPlaying = false;
    if (self.autoPlayTimer) {
      clearInterval(self.autoPlayTimer);
      self.autoPlayTimer = null;
    }
    self.dom.playBtn.innerHTML = 
      '<svg viewBox="0 0 24 24" width="18" height="18"><polygon points="8,4 20,12 8,20"/></svg> 自动播放';
    self.dom.playBtn.classList.remove('playing');
  };
  
  /**
   * autoGuide - 自动引导模式
   * 依次遍历所有步骤，每个停留2秒
   */
  self.autoGuide = function() {
    self.stopAutoPlay();
    self.currentStageIndex = 0;
    self.currentStepIndex = 0;
    self.updateDisplay();
    self.startAutoPlay();
  };
  
  /**
   * bindEvents - 绑定所有交互事件
   * 
   * 事件类型包括：按钮点击、键盘按键、窗口缩放
   * 
   * 底层原理：
   *   addEventListener 是浏览器原生的事件监听API
   *   第一个参数是事件名，第二个参数是回调函数
   *   回调函数在事件触发时由浏览器自动调用
   */
  self.bindEvents = function() {
    // 按钮点击事件
    self.dom.prevBtn.addEventListener('click', function() { 
      self.stopAutoPlay();
      self.prevStep(); 
    });
    self.dom.nextBtn.addEventListener('click', function() { 
      self.stopAutoPlay();
      self.nextStep(); 
    });
    self.dom.playBtn.addEventListener('click', function() { 
      self.toggleAutoPlay(); 
    });
    self.dom.autoGuideBtn.addEventListener('click', function() { 
      self.autoGuide(); 
    });
    
    // 键盘控制事件
    document.addEventListener('keydown', function(e) {
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        self.stopAutoPlay();
        self.nextStep();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        self.stopAutoPlay();
        self.prevStep();
      } else if (e.key === ' ') {
        e.preventDefault();
        self.toggleAutoPlay();
      }
    });
    
    // 窗口缩放事件
    window.addEventListener('resize', function() {
      self.updateDisplay();
    });
    
    // 阶段指示点点击事件
    for (var i = 0; i < self.dom.stageDots.length; i++) {
      (function(index) {
        self.dom.stageDots[index].addEventListener('click', function() {
          self.stopAutoPlay();
          self.currentStageIndex = index;
          self.currentStepIndex = 0;
          self.updateDisplay();
        });
      })(i);
    }
  };
};

// =================================================================
// 第四部分：页面加载启动
// 当浏览器完成HTML解析后，自动初始化教学引擎
// =================================================================
document.addEventListener('DOMContentLoaded', function() {
  var engine = new TeachingEngine();
  engine.init();
  
  // 暴露引擎到全局作用域，方便开发调试
  // 在浏览器控制台输入 window.teachingEngine 即可访问
  window.teachingEngine = engine;
  
  // 问题输入区域事件绑定
  var questionInput = document.getElementById('questionInput');
  var analyzeBtn = document.getElementById('analyzeBtn');
  
  if (questionInput && analyzeBtn) {
    // 点击分析按钮时分析问题
    analyzeBtn.addEventListener('click', function() {
      var questionText = questionInput.value.trim();
      if (questionText) {
        engine.configureForQuestion(questionText);
      } else {
        alert('请输入您想要学习的问题');
      }
    });
    
    // 按回车键时分析问题
    questionInput.addEventListener('keypress', function(e) {
      if (e.key === 'Enter') {
        var questionText = questionInput.value.trim();
        if (questionText) {
          engine.configureForQuestion(questionText);
        }
      }
    });
  }
});

/**
 * 常见问题排查：
 * 
 * 1. 图片不显示：
 *    - 检查图片URL是否可访问（浏览器F12→Network面板查看）
 *    - 替换为本地图片路径，如 "images/example.jpg"
 * 
 * 2. 光标位置不对：
 *    - 调整 TEACHING_STEPS 中对应步骤的 cursorX/cursorY 百分比值
 *    - 数值越大越靠右下角
 * 
 * 3. 闯关选项不显示：
 *    - 检查对应步骤的 isQuiz 字段是否为 true
 *    - 确保 quizOptions 和 quizAnswer 已正确配置
 * 
 * 4. 动画卡顿：
 *    - 减少 TEACHING_IMAGE_LIBRARY 中图片数量
 *    - 减小图片文件大小
 */
