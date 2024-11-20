## xPore 

AI ที่ตรวจหา RNA ภายใน Gnome

* Understanding domain knowledge
* Research questioning & requirement gathering
* Data exploration & preprocessing
* ML modeling
* Speed-up ML experiments
* Evaluation เช่น Generalized หรือ
* Deployment & documentation


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




## 2. Data Collection and Preparation

## 3. Bayesian (Multi-Sample) - Gaussian Mixture Modeling

## 4. Evaluation - 

## 5. Visualization and Presentation

โฆษณา ให้เขาเข้าใจว่าซอฟต์แวร์เราเป็นยังไง เพื่อนำไปใช้ในแต่ละแพลตฟอร์ม

## 6. Future Work
