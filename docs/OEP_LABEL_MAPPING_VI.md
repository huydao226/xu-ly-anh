# OEP Label Mapping (Tieng Viet)

## Muc dich

File nay ghi chu cach hieu cac nhan `type_x` cua OEP de dung cho:

- demo UI
- note bao cao
- giai thich su khac nhau giua model OEP va he thong heuristic cu

Can nhan manh:

- cac nhan `type_1` den `type_6` la nhan goc cua OEP hoac mirror dataset dang dung
- chung khong tuong duong 1-1 voi cac signal cua he thong cu nhu `head turn`, `looking down`, `phone`, `book`
- trong repo hien tai, chua co buoc map lai taxonomy OEP sang taxonomy rieng cua de tai

## Bang mapping de hieu nhanh

| Nhan | Dien giai tieng Viet | Do chac chan | Gan voi he thong cu |
| --- | --- | --- | --- |
| `normal` | Trang thai binh thuong, khong co doan cheating duoc gan nhan | Cao | `normal` |
| `type_1` | Xem sach, giay note, tai lieu giay | Cao theo paper OEP goc | Gan nhat voi `book` / `notes` |
| `type_2` | Noi chuyen voi nguoi khac trong phong | Cao theo paper OEP goc | Gan voi hanh vi nho nguoi khac ho tro, khong trung 1-1 voi signal cu |
| `type_3` | Su dung Internet | Cao theo paper OEP goc | He thong cu gan nhu khong quan sat truc tiep duoc bang 1 webcam |
| `type_4` | Hoi nguoi khac qua dien thoai | Cao theo paper OEP goc, nhung khong thay xuat hien trong mirror dataset hien tai | Co lien quan den `phone`, nhung khong trung 1-1 |
| `type_5` | Su dung dien thoai hoac thiet bi khac | Cao theo paper OEP goc | Gan nhat voi `phone detection` |
| `type_6` | Nhan xuat hien trong mirror dataset dang dung, chua co mo ta on dinh tu paper goc ma repo dang doi chieu | Thap | Chua nen map cung sang mot signal cu the |

## Giai thich quan trong

### 1. Tai sao UI moi hien `type_2` thay vi hien `phone` hoac `look away`

Model OEP moi duoc train theo nhan cua dataset OEP. Vi vay no tra ra:

- `normal`
- `type_1`
- `type_2`
- `type_3`
- `type_5`
- `type_6`

chu khong tra ra truc tiep:

- `phone`
- `book`
- `head turn`
- `looking down`

Ly do la model nay dang hoc `hanh vi theo nhan OEP`, con he thong cu la `detector + heuristic`.

### 2. So voi he thong cu, tuong dong den dau

He thong cu:

- nhin tung frame
- tim object va mot vai cue hinh hoc
- tra ra signal de giai thich ngay

Model OEP moi:

- nhin mot chuoi frame
- hoc pattern theo thoi gian
- tra ra nhan hanh vi cua OEP

Vi vay chi co the noi "gan voi", khong nen noi "bang nhau".

Mapping nen dung khi thuyet trinh:

- `type_1`: gan voi viec xem tai lieu
- `type_2`: gan voi viec tuong tac voi nguoi khac
- `type_3`: gan voi viec dung tai nguyen ngoai man hinh, nhung webcam 1 goc nhin khong thay ro ban chat
- `type_5`: gan voi viec dung dien thoai / thiet bi
- `type_6`: tam thoi de nguyen ma nhan, khong dien giai manh

### 3. Tai sao `type_6` can ghi chu rieng

Paper OEP goc thuong duoc mo ta voi 5 cheat types chinh.
Tuy nhien mirror dataset dang duoc import trong repo lai co:

- `type_1`
- `type_2`
- `type_3`
- `type_5`
- `type_6`

va khong co `type_4`.

Do do:

- khong nen tu y khang dinh `type_6` la mot hanh vi cu the neu chua doi chieu duoc nguon annotation goc
- an toan nhat la ghi: `type_6 la nhan xuat hien trong mirror dataset dang dung, can duoc xac minh lai neu muon dua vao bao cao chinh thuc`

## Cach viet an toan trong bao cao

Co the dung cau sau:

`Trong pipeline OEP, mo hinh temporal hien tai duoc huan luyen tren cac nhan hanh vi goc cua dataset, gom normal va mot so nhom cheating duoc danh dau bang ma type_1, type_2, type_3, type_5, type_6. Trong do, type_1 gan voi hanh vi xem tai lieu, type_2 gan voi hanh vi tuong tac voi nguoi khac, type_3 gan voi su dung Internet, type_5 gan voi dien thoai hoac thiet bi khac. Rieng type_6 can tiep tuc doi chieu annotation goc de dien giai chinh xac hon.` 

## Cach dat ten hien thi tren UI neu can than

Neu muon doi sang ten de demo ma van an toan:

- `normal` -> `Binh thuong`
- `type_1` -> `Nghiem van tai lieu`
- `type_2` -> `Tuong tac voi nguoi khac`
- `type_3` -> `Su dung tai nguyen ben ngoai`
- `type_5` -> `Su dung dien thoai / thiet bi`
- `type_6` -> `Hanh vi OEP khac`

Khong nen doi qua manh theo kieu:

- `type_2` -> `Noi chuyen`
- `type_3` -> `Len mang`
- `type_6` -> mot hanh vi cu the

neu chua co doi chieu annotation goc day du.

## Nguon doi chieu

- Paper OEP goc cua Michigan State University
- Manifest import trong repo:
  - `training/data/external/oep_multiview/notes/oep_webcam_segments.csv`
  - `training/models/oep_webcam_lstm_v2/label_map.json`

