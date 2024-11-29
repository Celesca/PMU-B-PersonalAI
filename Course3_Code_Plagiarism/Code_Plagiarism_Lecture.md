## AI for Code Clone Detection (MERRY)

1. What is Code Clones

![image](https://github.com/user-attachments/assets/104a40f3-86c7-4eb8-af17-b2a6bea474af)

แบ่งได้หลายชนิด 
* Type 1 : มีความเหมือนกันแบบเป๊ะๆ 
* Type 2 : เหมือนกันต่างกันแค่ตัวแปร
* Type 3 : อาจมีบางบรรทัดแก้ไข มีการตรวจสอบยาก เช่น return a + b; Refactoring เหลือบรรทัดเดียว
* Type 4 : ชนิดที่เพียงมีการคำนวณเหมือนกัน แต่ฟังก์ชันไม่เหมือนกัน (2 Method ไม่เหมือนกัน) ตรวจสอบยากที่สุด

![image](https://github.com/user-attachments/assets/b3a51efd-4de6-4475-b7b2-bd4f63f83d25)

ถ้ามี Source Code มันมีปัญหามากๆ มันจะ Maintain ได้ยาก ถ้าหากมันเกิด Bug

![image](https://github.com/user-attachments/assets/5cb1a17c-9071-44ee-8e91-188cf4f5e00c)

ปัญหาที่เจอในปัจจุบันก็คือ

1. เครื่องมือที่มีอยู่ใช้ยากต่อการตรวจจับ Several modifications
2. เครื่องมือ Clone detection กับ plagiarism detection ใช้ยากเพราะเป็น Command-line based tool

Objectives :

1. สร้างเครื่องมือตรวจ Code Clone Detection โดยใช้ Machine learning and study its effectiveness
2. เพื่อเพิ่ม user Experience ให้กับ Code Clone Detection Tools โดยทำเป็น เว็บแอพ กับ Visualization Clone Results

![image](https://github.com/user-attachments/assets/7ea27cf5-ceb2-46a0-b3ea-86ad4cf9dc67)


