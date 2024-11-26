## xPore: An AI-Powered App for Bioinformaticians 

In this module เราจะได้ Explore หัวข้อตามนี้ครับ

* Understanding domain knowledge
* Research questioning & requirement gathering
* Data exploration & preprocessing
* ML modeling
* Speed-up ML experiments
* Evaluation เช่น Generalized หรือ
* Deployment & documentation

---


## 1. Problem Statement (พัฒนาเพื่อแก้ปัญหาอะไร)

  ในงานวิจัยนี้ความรู้ที่เกี่ยวข้องเกี่ยวกับ Bioinformatics เช่น ATCG แปลงเป็น mRNA (Transcription)

   * 1.1 Central Dogma

![image](https://github.com/user-attachments/assets/a975de87-7e0f-4edc-ab77-c08d84838f9a)

เราจะดูยีนที่เป็นช่วงๆ แต่เราสนใจที่การแสดงออกของ Gene (Gene Expression) มันคือการ Transcription ออกมาเป็น mRNA
จำนวน Read ที่ออกมา

เราต้องรู้ว่า DNA mRNA ทำงานผิดปกติตรงไหน mRNA ที่มันออกมามากหรือน้อย ลำดับเบส (Sequencing) เอาเข้าเครื่องให้ดูลำดับเบส เพื่อดูทั้งสายว่าอันไหนแปลกไม่แปลก

   * 1.2 Past vs Current Solution

RNA มันค่อนข้าง Stable การ Convert CDNA เพื่อแปลงกลับ เราจะไม่สามารถ Detect ได้โดยตรง แต่ได้เป็น FastScript 

เราไม่จำเป็นต้อง CDNA เราใช้เทคโนโลยี Direct RNA Sequencing -> มันไม่ได้แปลงเป็น CDNA แล้วค่อย Sequencing แต่อันนี้มันทำได้โดยตรง

![image](https://github.com/user-attachments/assets/53b5ea14-5899-48a8-b8be-32cfba44bd43)

1. Directly RNA Sequencing
2. เครื่องเล็ก
3. ไฟล์ Fast5 แบบ Sequence Real-time

    * 1.3 RNA Sequencing Process

![image](https://github.com/user-attachments/assets/35414b95-51af-4556-9859-8f1d4ed805cc)

เนื่องจาก ลำดับเบสนั้น โมเลกุลมันอาจจะเปลี่ยนไป (RNA Modelification) เราเลยใช้เครื่องนี้ตรวจ
nanoPore Sequencer โปรตีนสองตัวมันเป็นเซนเซอร์สองขั้ว มันมีความต่างศักย์ไฟฟ้า หากมีโมเลกุลในนั้นจะมีความต้านทานที่แตกต่างกันออกไป ทำให้ส่งผลต่อค่าไฟฟ้า (Current)
ลำดับเบสแต่ละตัว โมเลกุล A หรือ T จะมี Pattern ของไฟฟ้าว่าแต่ละตัวมันคืออะไร เป็นสัญญาณลูกคลื่นขึ้นลง

Machine Learning ตัวนึงที่แปลงสัญญาณได้เรียกว่า Base calling (RNN, raw) -> มันเป็นเครื่องไฟฟ้าที่เป็นสัญญาณ
    * 1.4 RNA modifications
![image](https://github.com/user-attachments/assets/1cfa42d7-397e-4627-bd94-40dd097b0c72)


การเกิดการดัดแปลงเพื่อให้โปรตีนมันเป็นไปได้อย่างถูกต้อง นอกจาก Base AGCU แต่ละตัวมันดัดแปลงได้ ถ้าเปลี่ยนโดยตัว A ถ้ามี CH Group มาเกาะลำดับที่ 6 เรียกว่า m6A ที่ modify โดย CH Group
Common RNA ที่เจอได้มีส่วนปกติในร่างกาย เราไว้ศึกษา Splcing / RNA Instability / Translation / Disease-related
การที่มี Modification ที่เปลี่ยนจากปกติไปเป็น Modification

งานวิจัย Single-base-resolution CLIP-based detection methods
เขาสามารถทำ Lab ผลิต Enzyme ออกมา เวลาเตรียม Sample มันจะกินสาร RNA แต่จะกิน m6A ไม่ได้ ฉะนั้นเราจะรู้ตำแหน่ง m6A จากที่ถูกกิน เราเรียกว่า m6ACE-Seq

เราก็เลยคิดว่าเราควรจะใช้ nanoPore Sequencer มาใช้งานร่วมกันได้นอกจากการไปเตรียมแล็ป

  * 1.5 Datasets and Requirements Output Table

![image](https://github.com/user-attachments/assets/7340af7d-0f36-4d5e-98c1-8cbc5143e868)


nanoPore เวลามันบอกว่ามีสายไฟฟ้า มันไม่ได้บอกทีละ Base แต่มันจะเข้าไปเป็นกลุ่มของโมเลกุล 5 ตัว เรียกว่า 5-mer จากสัญญาณไฟฟ้า 5 ตัว เอามา Represent ตัวตรงกลางได้

Modify กับ ไม่ Modify มันเรียกว่า Modification rates differential of P-value ว่ามันเป็น Most significant รึเปล่า

KO (Knockout) - เอายีนที่ผลิตตัว m6A ออกไป ใน Sample เป็นการ Remove จาก RNA
WT (Wild Type) - มะเร็งเขา Modify น้อยมากแค่ไหน 

เราอยากรู้ Modification rates เป็นเท่าไหร่

  * 1.6 Research Objectives

![image](https://github.com/user-attachments/assets/1d450855-7964-4e41-8a21-7b93021d0280)

  ตอบโจทย์ว่าไม่มี Data ฉะนั้นเรา xPore ควรจะเป็น Unsupervised Learning แล้วก็ไม่เอา Training data


## 2. Data Collection and Preparation

จากเครื่องจะได้ไฟล์เป็น Fast5 (สัญญาณไฟฟ้า) แล้วแปลงจากสัญญาณไฟฟ้ามาเป็น
FastQ - เราจะเก็บลำดับเบสที่แล้ว Format เป็นคล้ายกับ Dictionary
FastA - เป็นลำดับเบส Reference Sequence บอกว่าลำดับเป็นหน้าตาเป็นยังไง มันเป็น Database แล้วก็มี Format เริ่มด้วย > ยีนอะไร แล้วก็หน้าตา CDNA
BAM/SAM - มันเก็บข้อมูล ผลลัพธ์ที่ Alignment ระหว่าง FastQ (ผลลัพธ์จาก Based-on) มา Align กับ FastA ( BAM - Binary / SAM - Text )
UCSC Genome Browser - Data Visualize ของลำดับ Base และ Histogram

การทำ Preprocessing จาก Nanopre เพื่อหา Signal-level data analysis
Direct RNA sequencing -> 
Basecalling (คือการคอลใช้ซอฟต์แวร์ มาจาก Deep Learning ว่าสัญญาณไฟฟ้าตัวไหนตัวตรงกลาง) โดยใช้ Guppy (FAST5 -> FASTQ) -> 
Sequence aligning Minimap2 (FASTQ + FASTA -> BAM) ผลของการ Align ว่า Read ไหนมันตรงกับ RNA ไหนใน Reference

![image](https://github.com/user-attachments/assets/27cee878-4f98-45b1-94f8-6d630fa14aa2)

Nanopolish มันจะเอาสัญญาณไฟฟ้ามาตรงกับ Base ไหน แล้วก็เอาผลลัพธ์ BAM มาดูยีนจะได้ Align ถูก มันก็จะเป็น Segment ของ Event Align Output เช่น ตรงนี้สัญญาณของ C T A G นะ แล้วจะเอาไปโมเดลอีกครั้ง


## 3. Bayesian (Multi-Sample) - Gaussian Mixture Modeling

![image](https://github.com/user-attachments/assets/454853a1-e90e-40bf-886d-cdc1b1b6ebf6)

เราใช้แบบ Bayesian GMM แล้วก็ให้มันลองรับ Multi-sample ด้วย 

3.1 Guassian คืออะไร 

แกน y คือ Probability แกน x เป็นสัญญาณไฟฟ้า ยิ่งกว้างมาก มันก็จะมีการกระจายตัวเยอะ ถ้าหากเรากระจายตาม Standard Deviation (Sigma)
สมมติเรามี Gaussian Distribution ขึ้นมา เราก็จะเห็นว่า Data จะเกาะกลุ่มที่ค่า Mean เราก็จะเขียน สมการ Probability(x) = N(x | มิว, Sigma)
ถ้าเราให้ มิวเป็น 0 Sigma เป็น 1 เราก็จะรู้ Probability ของจุดๆนั้นได้ทันที แล้วเราก็จะ Random Sample จุดแต่ละจุดจะได้ตาม Distribution / Probability ก็คือ สุ่ม นานๆทีจะเกิด

ข้อมูลไม่กระจายตัว Sigma σ จะแคบ ดูว่าโมเลกุลมันมีชนิดเท่าไหร่ Mean มีค่าเท่าไหน

![image](https://github.com/user-attachments/assets/532f6e81-d9c7-45a9-a7bc-4d0a11b2eaf4)

Mixture Model
ขั้นตอนการ Generate nanoPore มัน Assume ว่าสอง Distribution รวมกัน โดยที่สมการเปลี่ยนไปเป็นเพิ่ม k=1 ถึง K และ Pi K
Probability มันเป็นเท่าไหร่แล้วจะถูกเลือกเป็นสีส้มหรือสีน้ำเงิน เช่น Pi สีน้ำเงิน 90% Pi สีส้ม 10% แต่ถ้า Pi สีส้มเยอะกว่าก็จะอยู่สีส้ม

3.2 What is Gaussian Mixture Model (GMM) ?

![image](https://github.com/user-attachments/assets/340dc3e0-01db-4273-bd8f-e34dbc7e090c)


สัญญาณของไฟฟ้า Data ที่เราสร้างมาตามสมมติฐาน เช่น Data มี 3 Components และการกระจายตัวมากน้อยเป็น Normal Distribution 3 ชุด
เราก็ทำไป 5000 ครั้ง เช่นเราจะเอามาจากสีเขียว สีแดง สีน้ำเงิน มันก็จะได้รูปแบบนี้ โดยการ Random Components ว่าค่า Pi จะเป็นเท่าไหร่

ความเป็นจริง Machine Learning เราจะเห็น Data ก้อนเดียว แต่ไม่รู้มัน Generate ยังไง เราก็ Assume ว่าข้อมูลมาจาก Gaussian Mixture Model
การเห็นดาต้าเป็นสีชมพู แต่เราไม่รู้ Zeta ว่ามันไม่รู้มีกี่ Components, Mean , Sigma ของแต่ละอัน แต่ละ Component มีค่า Pi เท่าไหร่ แต่เรา
Assumption มันก่อนว่าเราใช้โมเดลอะไร ดาต้ามันถูกสร้างออกมาอย่างนั้น เราก็จะ Infer ว่า Pi, มิว, Sigma มีค่าเท่าไหร่ ก็ไป Learn

บางครั้งจุดมันเชื่อมกันเราก็แค่ให้มันเป็น Probability ไปก่อน

3.3 GMM Inference

![image](https://github.com/user-attachments/assets/b1176516-09c5-4ed4-aa99-c84501694442)

เริ่มต้นเราจะ Random Mean, Sigma กับการกระจายข้อมูลเป็นวงกลมก่อน ตอนแรกเรายังไม่รู้ว่ามันควรจะเป็นจุดไหน เราเลย Assign ว่าสีน้ำเงิน มันน่าจะมาจาก Probability นี้
แล้ว Iterative algorithm ทำการอัพเดทกลายเป็นวงกลมใหม่ จนสุดท้ายจะเจอ Mean, Sigma ที่ดีที่สุด ที่มัน Fits กับ Data ของเรามากที่สุด

3.4 ภาพรวมของ GMM

![image](https://github.com/user-attachments/assets/0c7a26b8-c30f-47a8-9232-397708120978)


GMM as a Density Estimator เป็น ML ที่เอาไว้ดู Distribution มีกี่อัน แล้วมีค่า Mean, Sigma เท่าไหร่ มันเอาไว้ประเมิน Density โดยมีการ Assumption ว่าข้อมูลมาจาก Normal Distribution
ซึ่งมันเรียกเป็น Generative AI เพราะว่าเรามี Assumption ว่าดาต้ามันมาจากไหน

Image Classification -> Discriminative มันไม่สามารถสร้างข้อมูลขึ้นมาได้

  1. มันสามารถสร้างข้อมูลได้ Generative AI -> มันสามารถ Random ว่าการสร้าง Data เพราะว่า Point มันมี Sample, Sampling , พอเรารู้ว่ามี Assumption
  2. เราก็สามารถทำ Clustering มันไม่มีค่า Mean, Representation ได้ แต่มันไม่สามารถ Generate ได้
  3. ยังมีคนทำ One-Class Classification กับการทำ Boundary

โมเดล / Data Generate ยังไง ก็คือจบที่โมเดล

3.5 Bayesian ( Learning algorithm - เอาไว้ประมาณค่าของ Zeta ว่า Mean, Sigma, Pi เป็นเท่าไหร่ )

![image](https://github.com/user-attachments/assets/8dc677f6-8cab-4cb6-b755-5c16ee06e1a7)


Point estimate (Prequentis) -> argmax มันคือ parameters ตัวไหนมีค่าเท่าไหร่ที่จะทำให้ Probability สูงที่สุด Data ทุกตัวมี Probability ทุกตัว ถ้ารวมกันแล้ว Parameter จะได้ Zeta ชุดเดียวที่สูงสุด
ปกติ Assumption ว่า k เท่าไหร่ เราก็มั่วมันไป เราไม่สามารถพล็อต Data เข้ามา ถ้าเราใช้ Point Estimate มันพยายามหา Distribution ของทุกๆ Components ตามที่เรากำหนด

Bayesian -> ไม่ได้ดูที่สูงสุด แต่อยากได้เป็น Distribution อย่างเดียว แต่ก็เป็นค่า Mean, บอก Uncertainty และ Prior จะเป็นยังไง
สำหรับการกำหนด K (Components) ในช่วงแรก คือการกำหนด Maximum k = 5 แต่ถ้าสมมติ มีแค่ 3 กลุ่ม แล้วสอง ไม่ Fits กับจุดใดๆ เลยก็ได้
ถ้ามันไม่มี Member อยู่ใน Dist มันก็จะไม่ตาย มันเป็น Prior (Believe) ที่ยังไม่เห็น Data

ข้อดีของ Point estimate -> พยายามหาจนกว่าจะครบ K = 5
ข้อดีของ Bayesian -> อนุญาตให้บาง Distribution ไม่มีจำนวนสมาชิกของมันเอง Data ที่มีอยู่อาจจะมีแค่สามก็ได้ ทำให้จุดที่อยู่ด้วยกันจริงๆ สองอันที่เหลือมันไม่ต้องใช้ได้

การเปรียบเทียบ Frequentist vs Bayesian

![image](https://github.com/user-attachments/assets/313b8862-9924-4e50-a7d7-9d72ac61e96d)

3.6 Multi-Sample

![image](https://github.com/user-attachments/assets/dcdcbd60-dd11-41ea-b473-594ae82aa937)


การเพิ่มว่ามันเป็นสีเดียว 2 Components เริ่มจากที่สีส้ม Sample 1 สีเขียวอาจจะมาจาก Sample 2 ทีนี้พอเอา RNA มา Sequence ก็จะเอาสัญญาณมาเลื่อนขึ้นลงกัน
ตรงจุดอื่นๆ สามสีอาจจะรวมกันเป็นเนื้อเดียว มีความแตกต่าง จะสังเกตว่า GGACT สีส้มมีค่า Mean ประมาณนี้ สีน้ำเงินมีประมาณต่ำกว่าอย่างเห็นได้ชัด สีเขียวมีบนและล่าง
1 เส้น 1 Read คนสีเขียวก็มีหลายๆ Read พอเราใช้ Histogram ขึ้นมา สิ่งที่เราจะทำคือ Fit Gaussian Mixture ลงไป แล้วพล็อตแยกออกมา
แล้วก็แยก Distribution เมื่อเรา Iterative Learn ไปเรื่อยๆ มันจะค่อยๆ พล็อต Distribution แยกว่าอันไหนมัน Fits กับ Data มากที่สุด

อันไหนมัน Modified คือไม่เคยเจอ (Unmodified current mean 90%, 3% modified current mean) เป็นการ Modification Rate ที่ออกมาได้
แล้ว Multi-sample เราจะ Compare กี่ Model ก็ได้

![image](https://github.com/user-attachments/assets/119f570b-58ab-49b8-9e69-12ed03180a28)

Assumption -> แต่ละไซส์และ Base มันก็คือบอกว่า M6A Modified กับ Label ไม่ Unmodified
เราก็เลือก K = 2 แล้วก็ขยาย Multi-sample
เราต้องใช้ Gaussian Distribution
เรารู้ Prior ว่า Unmodified k-mer เราก็ใส่ Prior
ซึ่งเราใช้ nanoPore Sequence เจ้าแรกที่ทำให้หาจาก Sequence ได้

ตำแหน่งไหนที่น่าจะตรงกับ Modification Rate มากที่สุด

3.7 Speed-Up ML Experiments

![image](https://github.com/user-attachments/assets/e354018a-fc83-4b01-b5b1-045dfcdb2fb1)


1. เราจะปรับ Hyper-parameter settings หรือการทดลองหลายสิบ Model ทำยังไง เพราะมีดาต้าเป็นล้าน เราก็จะสร้าง Configuration file ว่า
ค่าการ Experiment ตัวนี้มีค่าอะไร แล้วก็สร้าง Python packaging เอามาเรียกใช้ Model ง่าย
2. Parallelization
3. File indexing -> ไม่สามารถเอาขึ้นมาทั้ง RAM ได้

Why config files?

* Automating tasks , Centralised configuration, Documentation, Portability
* ที่เราใช้บ่อยคือ YML, JSON, TOML
* สามารถคอนฟิคได้ว่า Hyperparameters เป็นเท่าไหร่บ้าง

Python Packaging -> เอาไว้เรียกใช้งาน
Parallelization Multiprocessing -> multiprocessing หรือการอ่าน data.index ผ่าน data.json ก็จะอ่านเฉพาะตรงนั้นไฟล์ เราสามารถ Access ไฟล์ที่ใหญ่หลาย GB ได้แปปเดียว
Bioinformatics -> นิยมทำ Indexing

## 4. Evaluation - 

4.1 Experiment Setup

![image](https://github.com/user-attachments/assets/2da586b3-e51b-4537-8bac-bdf6b64a5451)

HEK293T Knockout -> มันมี Wild type, Knockout ยีนใน M6A มันจะถูกเอาออกหมดเลย ตามกระบวนการตามปฏิกิริยาเคมี
ตรงกลางต้องเป็น AC เท่านั้น ถึงจะนับว่า M6A

การวิเคราะห์แค่ Ground Truth จริงๆ มันถูกต้องไม่หมด 100% ฉะนั้น False Positives not be wrong มันน่าจะ Modify ตรงนั้นนะ ก็อาจจะไม่ได้ผิด แต่เป็นสมมติฐานที่ต้องไปพิสูจน์ต่อ
ว่าสายนี้จะเป็นไปได้จริง

Domain Specific Evaluation

![image](https://github.com/user-attachments/assets/4a314aed-bd84-4f02-8c16-d38e1c47c760)

Motis หลักๆเลยที่เป็น M6A ตามกราฟของ Motis Validate ว่ามันเป็นจริง ตรงกับ DARCH กับความรู้เก่า
และอีกอันนึงถ้าเราแบ่ง RNA 3' UTR มันจะเกิดตรงรอยต่อเยอะ เราพบว่าทุก ๆ ตำแหน่งที่ส่วนใหญ่มีพบมาก ทำให้ xPore มันคอนเฟิร์มความรู้เก่า

4.2 Validation ตำแหน่งที่มันใช่มันอยู่ตรงไหน เราจะเตรียม Sample ยังไง เราเชื่อว่า Sample เกิดจากการ KO และ Wild Type หมายความว่ามันมีกี่ %

![image](https://github.com/user-attachments/assets/6faa4194-62d7-4767-b2c7-979d4e335490)

เราต้องทำให้เขาดูว่ามันสามารถตีอะไรออกมาได้บ้าง Other Datasets ว่าตรงไหน ไต ตับ ตรงไหนมี M6A Modification Rate ออกมาเยอะ หรือเอา Clinical Data ว่าผู้ป่วยมัน Modify กันเท่าไหร่

![image](https://github.com/user-attachments/assets/ff960802-243a-47f2-b398-094f36209c2d)

Preprocess - 5-6 เดือน แต่ถ้า Evaluation ทำเป็นปี

4.3 Keys Takeaway

![image](https://github.com/user-attachments/assets/e7fc7dcc-d355-446f-a878-2bcc0fd501d6)

## 5. Visualization and Presentation (Deploy / Present Paper)
โฆษณา ให้เขาเข้าใจว่าซอฟต์แวร์เราเป็นยังไง เพื่อนำไปใช้ในแต่ละแพลตฟอร์ม

![image](https://github.com/user-attachments/assets/9288dcd3-8820-4220-8e0b-07c6040213dc)

เน้นเข้าใจง่าย Python ไม่สามารถจัดการได้โดยหมด เพื่อจะเล่าเรื่องราวเดียวกัน



## 6. Future Work
