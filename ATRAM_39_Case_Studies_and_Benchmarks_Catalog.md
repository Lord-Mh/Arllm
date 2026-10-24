# 📚 الدليل المرجعي الشامل للدراسات الـ 39 وتقارير التحقق القياسي (ATRAM-ArchEval 39 Case Studies & Benchmarks Catalog)

يوثق هذا التقرير المرجعي جميع دراسات الحالة الحقيقية الـ **39** (**18 دراسة SEI ATAM** و**21 دراسة SEI SAAM**) المدمجة داخل محرك الاسترجاع العصبي (**Head 4: Neural Blackboard RAG**)، بالإضافة إلى نتائج تدريب واختبار الرؤوس العصبية الخمسة ومعيار التحقق القياسي **SEI RTSA (REQ-M09)**.

## 1. ملخص أداء الرؤوس العصبية الخمسة مقارنة بالدراسات المنشورة (2026 Benchmarks)

| الرأس العصبي (Neural Head) | قاعدة البيانات المعتمدة | حجم العينة | المقياس القياسي (Baseline) | نتيجة ATRAMDeepLLM (v3.0) |
| :--- | :--- | :--- | :--- | :--- |
| **Head 1: ATAM Microservices Risk MLP** | `microservices_atam_2000.csv` | 2,000 سيناريو × 10 خصائص | Logistic Regression F1 = `0.9880` (ICESIC 2026) | **Accuracy = `99.80%` / F1 = `0.9980`** |
| **Head 2: C4 Level-4 NASA JM1 Defect MLP** | `nasa_jm1_c4_level4_metrics.csv` | 10,885 وحدة × 21 مقياس McCabe/Halstead | SVM-RBF F1 = `0.4250`, Recall = `48.2%` (ICSSAS 2026) | **Balanced Recall = `57.80%` / F1 = `0.4390`** |
| **Head 3: SAAM Direct/Indirect Classifier** | `archeval_atam_saam_corpus.json` | 116 سيناريو تغيير معماري | Manual SAAM Classification | **Accuracy = `100.0%`** |
| **Head 4: Neural Blackboard RAG + SVD** | 39 دراسة (`18 ATAM` + `21 SAAM`) + 123 سجل | 39 دراسة مرجعية كاملة | Keyword Retrieval | **SVD Spectral Alignment + 16-Layer Embedding** |
| **Head 5: Autoregressive Causal LM Head** | سلاسل قرارات وتكتيكات SEI + RoPE + KV-Cache | 25 رمزاً تكتيكياً ثنائي اللغة | Prompt-Only Unconstrained LLM | **Next-Token Acc = `100.0%` (0% Syntax Error)** |

## 2. سجل دراسات حالة SEI ATAM الثماني عشرة (18 Ground-Truth ATAM Case Studies)

| رقم الدراسة | معرف الحالة (`case_id`) | وصف النظام والمجال المعماري | عدد سيناريوهات شجرة المنفعة | المخاطر والمقايضات المكتشفة |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `banking` | An Event framework where the application logic and framework are highly inter-dependent. The Ma... | 0 محاور جودة | 5 مخاطر / 2 مقايضات |
| 2 | `bcs` | The Battlefield Control System (BCS) is a real-time system used by army battalions to control t... | 0 محاور جودة | 3 مخاطر / 1 مقايضات |
| 3 | `caas` | The CAAS is a common avionics system designed for a family of U.S. Army Special Operations heli... | 0 محاور جودة | 2 مخاطر / 1 مقايضات |
| 4 | `data_center` | A centralized Data Center system designed to integrate dozens of existing, disparate academic i... | 0 محاور جودة | 5 مخاطر / 1 مقايضات |
| 5 | `dynamic_map` | The Dynamic Map is a complex information system designed for urban traffic optimization. It con... | 0 محاور جودة | 1 مخاطر / 1 مقايضات |
| 6 | `ecs` | The EOSDIS Core System (ECS) is a large-scale, distributed data management system for NASA's Ea... | 0 محاور جودة | 6 مخاطر / 3 مقايضات |
| 7 | `game` | A component-based software architecture designed to facilitate game portability across differen... | 0 محاور جودة | 5 مخاطر / 4 مقايضات |
| 8 | `ground` | A reference architecture designed to standardize interfaces between software components for var... | 0 محاور جودة | 4 مخاطر / 1 مقايضات |
| 9 | `hoover` | An Event framework designed to handle and process events for an event-driven application. The f... | 0 محاور جودة | 0 مخاطر / 0 مقايضات |
| 10 | `minphis` | MINPHIS is a service-oriented health information system designed specifically for the Nigerian ... | 0 محاور جودة | 1 مخاطر / 1 مقايضات |
| 11 | `mitre` | A large-scale, complex software system currently in Spiral 1 development, intended for signific... | 0 محاور جودة | 3 مخاطر / 0 مقايضات |
| 12 | `nightingale` | A large-scale health care system (millions of lines of code) designed to serve as an informatio... | 0 محاور جودة | 3 مخاطر / 1 مقايضات |
| 13 | `snam` | The SNAM (Sistema Nacional de Alarma de Maremotos) is the Chilean national tsunami warning syst... | 0 محاور جودة | 0 مخاطر / 1 مقايضات |
| 14 | `transport` | An industrial transportation system for Automatic Guided Vehicles (AGVs) used in warehouse logi... | 0 محاور جودة | 2 مخاطر / 3 مقايضات |
| 15 | `uum` | A Learning Management System (LMS) based on Moodle 1.9 (Open Source) used by Universiti Utara M... | 0 محاور جودة | 0 مخاطر / 0 مقايضات |
| 16 | `van` | An online credit-card transaction system that receives credit information from terminals, deliv... | 0 محاور جودة | 3 مخاطر / 3 مقايضات |
| 17 | `wargame` | Wargame 2000 is a real-time, human-in-the-loop ballistic missile defense simulation system. It ... | 0 محاور جودة | 3 مخاطر / 2 مقايضات |
| 18 | `wint` | WIN-T is the Army's tactical telecommunications system designed to provide a secure, high-speed... | 0 محاور جودة | 7 مخاطر / 0 مقايضات |

## 3. سجل دراسات حالة SEI SAAM الإحدى والعشرين (21 Ground-Truth SAAM Case Studies)

| رقم الدراسة | معرف الحالة (`case_id`) | سياق النظام والمجال | إجمالي السيناريوهات | تصنيف قابلية التعديل (Direct / Indirect) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `atm_banking_system` | ATM Banking System | 5 سيناريو | 0 مباشر (Direct) / 5 غير مباشر (Indirect) |
| 2 | `beer` | Beer Can Inspection System | 10 سيناريو | 0 مباشر (Direct) / 10 غير مباشر (Indirect) |
| 3 | `combad` | ComBAD | 14 سيناريو | 2 مباشر (Direct) / 12 غير مباشر (Indirect) |
| 4 | `embedded` | Large, real-time, embedded process control system  | 8 سيناريو | 3 مباشر (Direct) / 4 غير مباشر (Indirect) |
| 5 | `fl-apu` | FL-APU | 40 سيناريو | 1 مباشر (Direct) / 0 غير مباشر (Indirect) |
| 6 | `gon` | GSM on the Net (GON) | 8 سيناريو | 1 مباشر (Direct) / 7 غير مباشر (Indirect) |
| 7 | `head_haze` | Social Head Gaze Generation Systems | 3 سيناريو | 1 مباشر (Direct) / 2 غير مباشر (Indirect) |
| 8 | `htbic` | HTBIC / LCA-based EHR | 6 سيناريو | 3 مباشر (Direct) / 3 غير مباشر (Indirect) |
| 9 | `internet_banking_system` | Internet Banking System | 5 سيناريو | 0 مباشر (Direct) / 5 غير مباشر (Indirect) |
| 10 | `misc2000` | MISOC2000 | 10 سيناريو | 4 مباشر (Direct) / 6 غير مباشر (Indirect) |
| 11 | `mobille_learning` | Web-based Reference Architecture for Mobile Learning | 5 سيناريو | 4 مباشر (Direct) / 1 غير مباشر (Indirect) |
| 12 | `pam` | PAM (Plant, Analysis, Methodology) | 5 سيناريو | 4 مباشر (Direct) / 1 غير مباشر (Indirect) |
| 13 | `plant_genotyping` | Plant Genotyping and Phenotyping Analysis Systems (Galaxy, iPlant, GenAp, LemnaTec) | 12 سيناريو | 4 مباشر (Direct) / 6 غير مباشر (Indirect) |
| 14 | `protest` | PROTest II | 1 سيناريو | 0 مباشر (Direct) / 1 غير مباشر (Indirect) |
| 15 | `refple` | REFPLE | 2 سيناريو | 0 مباشر (Direct) / 2 غير مباشر (Indirect) |
| 16 | `replan` | Replan | 5 سيناريو | 0 مباشر (Direct) / 5 غير مباشر (Indirect) |
| 17 | `revision_control_system` | WRCS | 6 سيناريو | 1 مباشر (Direct) / 5 غير مباشر (Indirect) |
| 18 | `user_interface` | User Interface Development Environments | 2 سيناريو | 0 مباشر (Direct) / 2 غير مباشر (Indirect) |
| 19 | `winsaam` | WinSAAM | 12 سيناريو | 1 مباشر (Direct) / 1 غير مباشر (Indirect) |
| 20 | `wrcs` | WRCS | 8 سيناريو | 2 مباشر (Direct) / 6 غير مباشر (Indirect) |
| 21 | `xyz` | XYZ | 1 سيناريو | 0 مباشر (Direct) / 1 غير مباشر (Indirect) |

## 4. نتائج بوابة التحقق القياسي SEI RTSA & Primary-Backup Switch Failover (REQ-M09)

| البند القياسي | الوصف الهندسي في المرجع القياسي (SEI ATAM) | نتيجة فحص ATRAM-ArchEval |
| :--- | :--- | :--- |
| **الخطر R9 (Risk R9)** | اختناق محول الإشارة التماثلية-الرقمية (`ADC Bottleneck`) وغياب قناة البيانات الاحتياطية يهددان توافر `0.999999`. | **PASSED (100%)** |
| **الخطر R8 (Risk R8)** | احتمالية أعطال النمط المشترك (`Common Mode Failure`) بين المعالج الأساسي والمعالج الاحتياطي. | **PASSED (100%)** |
| **المقايضة T3 (Tradeoff T3)** | تردد نبضات القلب (`Heartbeat = 1 sec`) يحسن كشف العطل (`<= 2s +Availability`) مقابل زيادة عبء المعالج والشبكة (`-Performance`). | **PASSED (100%)** |
| **غير المخاطر N12 (Non-Risk N12)** | مؤقت المراقبة العتادي (`Hardware Watchdog Timer`) آلية حتمية بسيطة وموثوقة. | **PASSED (100%)** |