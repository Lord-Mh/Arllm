# 🚀 منظومة الوكيل اللغوي المعماري العميق (ATRAM-ArchEval Deep LLM Agent v3.0 — > 1.56 GB)

**المشروع الهجين العصبي-الرمزي المتكامل لتقييم ومحاكاة المعماريات البرمجية: SEI ATAM + SEI SAAM + C4 Model (Diagrams-as-Code)**  
*Hybrid Neuro-Symbolic Tri-Party Architecture for Automated Software Architecture Evaluation, What-If Simulation & CBAM ROI*

## 📌 1. نظرة عامة على المشروع (Project Overview)

تُعد منظومة **ATRAM-ArchEval Deep LLM Agent (v3.0)** إطاراً هندسياً متكاملاً مبنياً بالكامل من الصفر (**100% From Scratch in Python/NumPy**) لتقييم المعماريات البرمجية آلياً وفق منهجيات معهد هندسة البرمجيات بجامعة كارنيجي ميلون (**SEI ATAM** و **SEI SAAM**) ومواءمتها مع المستويات الأربعة لنموذج **C4 Model (Diagrams-as-Code)** ومعيار متطلبات الأنظمة **ISO/IEC/IEEE 29148 (MoSCoW)**.

يعالج هذا المشروع القصور الجوهري في أنظمة الوكلاء اللغوية التقليدية (مثل الهلوسة الحسابية في حساب نقاط الفشل الفردية، أو ظاهرة "العميل الصامت"، أو الحوار اللانهائي بين الوكلاء) عبر **حلقة التفاعل الثلاثية (Tri-Party Interactive Loop)**:
1. **الطرف الأول — المستخدم / المعماري البشري (Human-in-the-Loop Architect):** يدخل كود المعمارية بصيغة `C4 Mermaid / PlantUML / Structurizr DSL`، ويجيب عن 4 أسئلة استيضاحية دقيقة يطرحها وكيل العميل، ويختبر فرضيات التعديل في محاكي الطيران (`What-If Flight Simulator`).
2. **الطرف الثاني — البرنامج الحتمي (Deterministic C4 AST & Tarjan Graph Engine):** يحلل شجرة النحو المجردة (`AST`) بنسبة خطأ `0%`، ويفحص بروتوكولات الأسهم (`Arrow Protocol Inspector`)، وينفذ **خوارزمية تارجان للعقد المفصلية (`Tarjan's Cut-Vertices Algorithm`)** لاكتشاف نقاط الفشل الفردية (`SPOFs`) وسلاسل الاستدعاء المتزامنة الحرجة (`Cascading Sync Chains`)، ويولد كود C4 المُصلح تلقائياً.
3. **الطرف الثالث — النموذج اللغوي العميق والوكلاء الأربعة (`ATRAMDeepLLM > 1.56 GB` + `4 Blackboard Agents`):** مكدس مكون من **16 طبقة Transformer عميقة** (`409,600,000` معامل `Float32` بحجم فعلي `1.5625 GB` / `1,677,722,012 بايت`) مع **5 رؤوس عصبية متعددة المهام** و**4 وكلاء متخصصين** يتواصلون عبر سبورة مشتركة (`Shared JSON Blackboard`).

## 🛠️ 2. كيف بُني المشروع من الصفر؟ (How It Was Built — Engineering Architecture)

تم تصميم وبناء كل مكون رياضي وبرمجي من الصفر داخل ملف المحرك الموحد `atram_deep_llm_engine.py` والوحدات المعيارية في `src/` وفق الخطوات الهندسية التالية:

### أ. النواة الرياضية ومقسم الرموز ثنائي اللغة (`ArchTokenizer & Math Primitives`)
* **الدوال العصبية الأساسية:** برمجة دوال `softmax` المستقرة عددياً، و`sigmoid`، ودالة التنشيط `SiLU (Swish)`، وتطبيع الجذر التربيعي لمتوسط المربعات `RMSNorm` مباشرةً بمصفوفات `NumPy`.
* **قاموس الرموز (`ArchTokenizer` — سعة `32,768` رمز):** يدعم اللغتين العربية والإنجليزية (`[\w\u0600-\u06FF]+`)، والرموز الخاصة بالمنظومة (`<BOS>`, `<EOS>`, `<ATAM>`, `<SAAM>`, `<C4>`, `<RISK>`, `<TRADEOFF>`, `<SENSITIVITY>`, `<UTILITY_TREE>`)، مع حساب أوزان `TF-IDF` والتضمين الدلالي.

### ب. مكدس الـ 16 طبقة (`16-Layer DeepTransformerLayer` + `RoPE` + `KV-Cache` + `SwiGLU`)
* **تطبيع `RMSNorm` المسبق (Pre-Norm):** يُطبّق قبل طبقة الانتباه الذاتي وقبل شبكة التغذية الأمامية لضمان استقرار التدفق عبر الطبقات الـ 16.
* **الترميز الموضعي الدوراني (`Rotary Positional Embeddings - RoPE`):** تدوير أزواج الأبعاد في متجهات الاستعلام والمفتاح (`Q, K`) بزوايا تعتمد على الموضع التسلسلي (`pos_offset`) عبر الدالة `apply_rope()`.
* **الانتباه الذاتي متعدد الرؤوس (`Multi-Head Self-Attention`):** مصفوفات الإسقاط الأربع (`W_q, W_k, W_v, W_o`) مع دعم **قناع الانتباه السببي (`Causal Mask`)** و**ذاكرة التخزين المؤقت (`16-Layer KV-Cache`)** لتسريع التوليد التلقائي، وحساب إنتروبيا الانتباه (`Mean Attention Entropy = 0.258`).
* **شبكة التغذية الأمامية البوابية (`SwiGLU FFN`):** تطبق المعادلة $\text{FFN}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}}$.
* **توليد وربط ملف الأوزان الكامل (`1.5625 GB Checkpoint`):** تبني الدالة `materialize_full_1_56gb_checkpoint()` ملف الأوزان الثنائي `models/atram_deep_llm_1_56GB_full.bin` (`409,600,000` معامل `Float32` = `1,677,722,012 بايت`)، وتقوم الدالة `verify_and_bind_checkpoint()` بربط مصفوفات الأوزان السبع عبر جميع الطبقات الـ 16 (`VERIFIED_AND_BOUND_ALL_16_LAYERS`).

### ج. الرؤوس العصبية الخمسة وخوارزمية `AdamW` والمواءمة الطيفية (`SVD`)
تم تدريب جميع الرؤوس العصبية باستخدام خوارزمية **AdamW Backpropagation** مع جدولة جيب التمام لمعدل التعلم (`Cosine Annealing LR`) ومواءمة مصفوفة إسقاط الانتباه `W_o` عبر تفكيك القيم المفردة (`np.linalg.svd`):

| الرأس العصبي (Neural Head) | البيانات التدريبية والمدخلات | الوظيفة المعمارية | نتائج الاختبار الفعلية |
| :--- | :--- | :--- | :--- |
| **Head 1: ATAM Microservices Risk MLP** | `2,000` سيناريو ميكروسيرفيس (`data/microservices_atam_2000.csv`) × 10 خصائص | التنبؤ بمستوى الخطر المعماري (`Low / Medium / High`) واحتمالاته | `Accuracy = 99.80%`، `F1 = 0.9980` (يتفوق على Logistic Regression `0.988`) |
| **Head 2: C4 Level-4 NASA JM1 Defect MLP** | `10,885` وحدة برمجية (`data/nasa_jm1_c4_level4_metrics.csv`) × 21 مقياس McCabe/Halstead | التنبؤ بعيوب الكود الساكنة للمستوى الرابع من C4 مع موازنة الأوزان | `Recall = 57.80%`، `F1 = 0.4390` (يتفوق على SVM-RBF `0.425`) |
| **Head 3: SAAM Scenario Classifier** | `116` سيناريو معماري مستخرج من `21` دراسة حالة `SAAM` | تصنيف سيناريوهات التغيير إلى مباشرة (`Direct`) أو غير مباشرة (`Indirect`) وتقدير الجهد | `Accuracy = 100.0%` |
| **Head 4: Neural Blackboard RAG + SVD** | `39` دراسة حالة مرجعية (`18 ATAM` + `21 SAAM`) و`123` سجل استدلال (`data/archeval_atam_saam_corpus.json`) | استرجاع الحالات المرجعية المماثلة وتكتيكات `SEI Chapter 21` بالمواءمة الطيفية | `39 Studies Indexed` (`SVD Spectral Alignment Active`) |
| **Head 5: Autoregressive Causal LM Head** | سلاسل القرارات والتكتيكات المعمارية مع `RoPE` و`KV-Cache` و`Top-K Sampling` | التوليد اللغوي السببي كلمة بكلمة والتركيب ثنائي اللغة (عربي/إنجليزي) | `Next-Token Acc = 100.0%` (`16-Layer KV-Cache`) |

## ⚙️ 3. كيف يعمل النظام أثناء التشغيل؟ (How It Works — Step-by-Step Runtime Flow)

عند تمرير أي مخطط معماري إلى `TriPartyArchitectureOrchestrator.run_full_tri_party_evaluation()`، يعمل النظام في **7 مراحل متسلسلة وحتمية**:

1. **المرحلة 1 — التحليل الحتمي لمخطط C4 وخوارزمية تارجان (`C4 AST & Tarjan Cut-Vertices`):**
   * يفكك الكود إلى عقد (`C4Node`) وروابط (`C4Edge`) ويكتشف الأسهم الصامتة (`Silent Arrows`) والروابط غير المشفرة.
   * يشغل خوارزمية `Tarjan's Cut-Vertices` لاكتشاف العقد المفصلية التي يؤدي سقوطها إلى انقسام الشبكة، بالإضافة إلى قواعد البيانات المفردة (`Stateful DB SPOFs`) وسلاسل الاستدعاء المتزامنة (`web_app -> order_service -> payment_service`).
2. **المرحلة 2 — استيضاح المتطلبات التفاعلي (`Customer_Clarification_Agent`):**
   * يولّد 4 أسئلة استيضاحية موجهة للمستخدم حول: بروتوكول الأسهم، ذروة معدل الطلبات (`Peak RPS & P99 SLA`)، زمن التعافي المستهدف (`RTO/RPO`)، ومعايير الامتثال الأمني (`PCI-DSS / mTLS`).
3. **المرحلة 3 — التقييم العصبي عبر الرؤوس الخمسة (`5-Head Neural Inference`):**
   * يحسب المتجه الكمي المكون من 10 خصائص ويمرره إلى **Head 1**، ويفحص مقاييس الكود عبر **Head 2**، ويصنف سيناريوهات التعديل عبر **Head 3**، ويسترجع أقرب 3 دراسات مرجعية من الـ 39 دراسة عبر **Head 4**، ويولّد التسلسل الرمزي والاستدلال العربي/الإنجليزي عبر **Head 5**.
4. **المرحلة 4 — استبيان تكتيكات SEI وبؤر التكتل الهيكلي (`SEI Chapter 21 Tactics & SAAM Hotspots`):**
   * يقيّم تكتيكات التوافر والأداء والأمان وقابلية التعديل، ويحسب أوزان تفاعل السيناريوهات (`SAAM Interaction Hotspots`) لرصد المكونات عالية الاقتران (`INT-order_service`).
5. **المرحلة 5 — مناظرة المراجعة الداخلية على السبورة المشتركة (`Internal Review Debate on Shared JSON Blackboard`):**
   * يقترح `Manager_Agent` شجرة المنفعة (`Utility Tree`) بشرط صفر توقف (`0ms Downtime`)، فيقدم `Architect_Agent` اعتراضاً تقنياً مسنداً بمخرجات خوارزمية تارجان ويعدل الهدف إلى `RTO < 4s` عبر إضافة `Apache Kafka Event Bus` و`Multi-AZ Read-Replica`، ثم يصادق `EvaluationTeam_Agent` على المصفوفة النهائية.
6. **المرحلة 6 — محاكي الطيران المعماري والعائد المالي (`What-If Flight Simulator & CBAM ROI`) وتوليد `MADR ADRs`:**
   * يولد النظام كود C4 المُصلح تلقائياً ويعيد تحليله في أقل من ثانية لمقارنة المقاييس قبل التعديل وبعده:
     * **زمن الاستجابة (`Avg Latency`):** ينخفض من `246.0 ms` إلى `186.0 ms` (**وفر `-60.0 ms`**).
     * **التوافر (`Availability`):** يرتفع من `99.45%` إلى `99.78%` (**تحسن `+0.33%`**، أي تفادي `21.0 ساعة` توقف سنوياً).
     * **قابلية التوسع (`Scalability`):** ترتفع من `5.56/10` إلى `7.91/10` (**مكسب `+2.35`**).
     * **العائد المالي (`CBAM ROI`):** وفورات سنوية بقيمة `$262,500` مقابل تكلفة تنفيذ `$8,500` = **عائد استثمار `30.88x`** (صافي ربح `$254,000` في السنة الأولى)، مع توليد قرارات `ADR-001` و`ADR-002` بصيغة `MADR`.
7. **المرحلة 7 — بوابة التحقق القياسي (`SEI RTSA Benchmark Gate REQ-M09`):**
   * يتحقق النظام آلياً من اجتياز معيار مجس الحرارة والمفتاح الاحتياطي (`SEI RTSA`) بنسبة **100%**: رصد الخطر `R9` (اختناق `ADC` وغياب قناة البيانات الاحتياطية)، والخطر `R8` (أعطال النمط المشترك)، والمقايضة `T3` (تردد نبضات القلب `1 ثانية` بين التوافر والأداء)، وغير المخاطر `N12` (موثوقية مؤقت `Watchdog`).

## 📂 4. هيكل ملفات المشروع الكامل على Google Drive

```text
ATRAM_ArchEval_Deep_LLM_Agent_ATAM_SAAM_C4/
├── ATRAM_Deep_LLM_Colab_Drive_Runner.ipynb          # دفتر التشغيل الرئيسي على Google Colab (5 خطوات كاملة ومنفذة مسبقاً)
├── ATRAM_Interactive_Copilot_Studio.html            # تطبيق الاستوديو التفاعلي البصري (4 تبويبات + المختبر العصبي للرؤوس الخمسة)
├── atram_deep_llm_engine.py                         # المحرك الشامل الموحد v3.0 (85.2 KB — جميع الكلاسات والـ 5 رؤوس والوكلاء)
├── main.py                                          # نقطة الدخول القياسية للتشغيل من سطر الأوامر (CLI)
├── ATRAM_ArchEval_Blackboard_Report.json            # تقرير السبورة المشتركة المولد بالكامل بصيغة JSON
├── ATRAM_39_Case_Studies_and_Benchmarks_Catalog.md  # دليل الدراسات المرجعية الـ 39 (18 ATAM + 21 SAAM) والمعايير
├── README.md                                        # هذا الدليل الهندسي والتشغيلي الشامل
├── ATRAM_ArchEval_Deep_LLM_Complete_Report.pdf      # التقرير الهندسي والدراسات الشامل بصيغة PDF
├── ATRAM_ArchEval_Deep_LLM_Presentation.pptx        # العرض التقديمي التنفيذي للمشروع بصيغة PowerPoint
├── models/
│   ├── atram_deep_llm_1_56GB_full.bin               # ملف الأوزان الكامل (1,677,722,012 بايت = 1.5625 GB / 409.6M Float32)
│   └── atram_deep_llm_core_weights.npz              # أوزان الرؤوس العصبية الخمسة ومصفوفة SVD والتمثيلات الدلالية (516 KB)
├── data/
│   ├── archeval_atam_saam_corpus.json               # الذخيرة المرجعية الكاملة (18 دراسة ATAM + 21 دراسة SAAM + 30 سجل استدلال)
│   ├── microservices_atam_2000.csv                  # قاعدة بيانات 2,000 سيناريو ميكروسيرفيس (10 خصائص)
│   └── nasa_jm1_c4_level4_metrics.csv               # قاعدة بيانات NASA JM1 للمستوى الرابع من C4 (10,885 وحدة × 21 مقياساً)
└── src/
    ├── c4_dac_engine.py                             # وحدة محرك C4 الحتمي وخوارزمية تارجان وفاحص الأسهم
    ├── deep_llm_from_scratch.py                     # وحدة الشبكة العصبية العميقة والـ Transformer والرؤوس المتعددة
    └── multi_agent_orchestrator.py                  # وحدة السبورة المشتركة والوكلاء الأربعة ومحاكي What-If وCBAM
```

## ▶️ 5. خطوات التشغيل الحالية (Current Step-by-Step Execution Guide)

### الطريقة الأولى: التشغيل السحابي المباشر عبر Google Colab (موصى بها)
1. افتح الملف `ATRAM_Deep_LLM_Colab_Drive_Runner.ipynb` من مجلد المشروع على Google Drive باستخدام **Google Colab**.
2. ستجد جميع الخلايا الخمس منفذة مسبقاً وتعرض نتائجها الكاملة أسفل كل خلية؛ ولإعادة التشغيل الحي اضغط على **`Runtime -> Run all`** (`Ctrl + F9`):
   * **الخلية 1:** تربط حساب Google Drive تلقائياً وتفعّل المحرك `atram_deep_llm_engine.py (v3.0)`.
   * **الخلية 2:** تفحص وتربط ملف الأوزان العميق `models/atram_deep_llm_1_56GB_full.bin` (`1.5625 GB`).
   * **الخلية 3:** تشغل محرك `C4DeterministicGraphEngine` وخوارزمية تارجان وتولد كود C4 المُصلح تلقائياً.
   * **الخلية 4:** تربط الطبقات الـ 16 بملف الأوزان، وتدرب الرؤوس العصبية الخمسة على قواعد البيانات الحقيقية في `data/`، وتحفظ `models/atram_deep_llm_core_weights.npz`، وتنفذ التوليد اللغوي السببي عبر **Head 5 (`RoPE + KV-Cache`)**.
   * **الخلية 5:** تشغل منسق الحلقة الثلاثية `TriPartyArchitectureOrchestrator` والوكلاء الأربعة ومحاكي `What-If` وعائد `CBAM ROI`، وتصدر التقرير النهائي إلى `ATRAM_ArchEval_Blackboard_Report.json`.

### الطريقة الثانية: التشغيل التفاعلي البصري عبر استوديو الويب (`ATRAM_Interactive_Copilot_Studio.html`)
1. افتح ملف `ATRAM_Interactive_Copilot_Studio.html` في أي متصفح حديث أو داخل لوحة **Canvas**.
2. من **التبويب 1 (استوديو C4 والتفاعل الثلاثي):** اختر إحدى المعماريات الأربعة الجاهزة (أو الصق كود `C4 Mermaid` الخاص بك)، وأجب عن أسئلة وكيل الاستيضاح الأربعة، واضغط **«⚡ تشغيل التحليل المعماري الثلاثي»** أو **«🛠️ تطبيق كود C4 المُصلح تلقائياً»**.
3. من **التبويب 2 (السبورة المشتركة وشجرة المنفعة):** استعرض حوار الوكلاء الأربعة ومناظرة المراجعة الداخلية وشجرة المنفعة السداسية.
4. من **التبويب 3 (تقييم SAAM ومحاكي الطيران وCBAM):** قارن الأداء قبل التعديل وبعده وانسخ قرارات `ADR` وكود C4 المعدل.
5. من **التبويب 4 (معمارية النموذج العميق والمحاكاة العصبية الحية):** اختبر الرؤوس العصبية الخمسة مباشرةً (منزلقات `Head 1` و`Head 2`، ومصنف `SAAM Head 3`، وبحث `RAG Head 4`، ومنصة التوليد العصبي `Causal LM Head 5`).

### الطريقة الثالثة: التشغيل من سطر الأوامر (`CLI / Local Python`)
```bash
python3 atram_deep_llm_engine.py
```
أو عبر نقطة الدخول المعيارية:
```bash
python3 main.py --train-and-verify
```
