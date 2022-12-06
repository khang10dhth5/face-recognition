import os
import threading
from  PyQt5.QtWidgets import QApplication,QMessageBox, QMainWindow
from  PyQt5 import  uic,QtWidgets,QtCore,QtGui
from  PyQt5.QtCore import QTimer
from  cv2 import cv2
import  numpy as np
from PIL import Image
import  sys
import pyodbc
import time
import anhSV
import  pyshine as ps
from anhSV import frm_AnhSV
import tensorflow as tf
import keras
from PyQt5.QtCore import QTimer
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.layers import BatchNormalization
from keras.models import load_model
from keras_preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from  cv2 import cv2
from  threading import  Timer,Thread
from numpy import expand_dims
from mtcnn_cv2 import MTCNN
from keras.models import load_model
import pickle

con=pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server}; SERVER=NITRO5;Database=KhoaLuan;UID=sa;PWD=123')
cursor=con.cursor()
face_detector = MTCNN()
MyFaceNet = load_model("facenet_keras.h5")
class MAIN(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("main.ui",self)
        self.setFixedSize(1519,742)
        self.show()
        self.btnSinhVien.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_SinhVien))
        self.btnMonHoc.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_MonHoc))
        self.btnDiemDanh.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_LichDay))
        self.btnGiangVien.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_GiangVien))

        timer = QTimer(self)
        timer.timeout.connect(self.GetThoiGian)
        timer.start(100)
        ######################################################################################################
        # page môn học
        self.themlhp=False
        self.themmh=False
        self.loadMonHoc()
        self.tbl_MonHoc_2.clicked.connect(self.GetMonHoc)
        self.tbl_LopHocPhan_2.clicked.connect(self.GetLopHocPhan)
        self.tbl_DSSV_2.clicked.connect(self.GetSV_MonHoc)
        self.KhoiTaoMH()
        self.btn_ThemMH_2.clicked.connect(self.ThemMH)
        self.btn_HuyLopHoc_2.clicked.connect(self.HuyMH)
        self.btn_SuaMH_2.clicked.connect(self.SuaMH)
        self.btn_LuuMH_2.clicked.connect(self.LuuMH)
        self.btn_XoaLopHoc_2.clicked.connect(self.XoaMH)
        self.btn_ThemLopHP_2.clicked.connect(self.ThemLHP)
        self.btn_HuyLHP_2.clicked.connect(self.HuyLHP)
        self.btn_SuaLopHP_2.clicked.connect(self.SuaLHP)
        self.btn_LuuLHP_2.clicked.connect(self.LuuLHP)
        self.btn_XoaLopHP_2.clicked.connect(self.XoaLHP)
        self.btn_TimSV_2.clicked.connect(self.TimSVThemLop)
        self.btn_ThemSV_2.clicked.connect(self.ThemSVVaoLop)
        self.btn_XoaSV_2.clicked.connect(self.XoaSVKhoiLop)
        self.btn_TimMH_2.clicked.connect(self.TimMonHoc)
        self.btn_AllMH_2.clicked.connect(self.AllMH)
        # self.btn_DatLichHoc_2.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_LichHoc))
        self.btn_DatLichHoc_2.clicked.connect(self.DatLichHoc)
        self.btn_BackPageMH_4.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_MonHoc))
        ######################################################################################################
        #page sinh viên
        self.anhSV = frm_AnhSV()
        self.loadData()
        self.KhoiTao()
        self.KhoiTaoCombobox()
        self.btn_LayAnhSV.clicked.connect(self.LayAnh)
        self.them = False
        self.themlop = False
        self.tbl_SinhVien.clicked.connect(self.GetSV)
        self.tbl_Lop.clicked.connect(self.GetLop)
        self.btn_AllSV.clicked.connect(self.GetAllSV)
        self.btn_TimSV.clicked.connect(self.TimSV)
        self.btn_TimLop.clicked.connect(self.TimLop)
        self.btn_ThemSV.clicked.connect(self.themSV)
        self.btn_LamMoiSV.clicked.connect(self.LamMoiSV)
        self.btnLuuSV.clicked.connect(self.LuuSV)
        self.btnHuySV.clicked.connect(self.HuySV)
        self.btn_SuaSV.clicked.connect(self.SuaSV)
        self.btn_XoaSV.clicked.connect(self.XoaSV)
        self.btn_ThemLop.clicked.connect(self.ThemLop)
        self.btn_SuaLop.clicked.connect(self.SuaLop)
        self.btn_HuyLop.clicked.connect(self.HuyLop)
        self.btn_LuuLop.clicked.connect(self.LuuLop)
        self.btn_XoaLop.clicked.connect(self.XoaLop)
        self.btn_TrainingData.clicked.connect(self.Training)
        self.progressBar.setValue(0)
        self.progressBar.setVisible(False)
        ######################################################################################################
        # page buổi học
        self.KhoiTaoBH()
        self.btn_ThemBH_4.clicked.connect(self.ThemBH)
        self.thembh=False
        self.btn_HuyBH_4.clicked.connect(self.HuyBH)
        self.btn_SuaBH_4.clicked.connect(self.SuaBH)
        self.tbl_DSBuoiHoc_4.clicked.connect(self.GetBH)
        self.btn_LuuBH_4.clicked.connect(self.LuuBH)
        self.btn_XoaBH_4.clicked.connect(self.XoaBH)
        self.btn_ChiTietDD_4.clicked.connect(self.ChiTietDD)
        ######################################################################################################
        # page lịch dạy
        self.btn_DiemDanh_6.clicked.connect(self.OpenDiemDanh)
        self.loadThoiKhoaBieu()
        self.tbl_DSLopHP_6.clicked.connect(self.GetLopHPDD)
        self.tbl_DSBuoiHoc_6.clicked.connect(self.GetBuoiHocDD)
        ######################################################################################################
        # page điểm danh
        self.btn_MoCam_3.clicked.connect(self.OpenCamDiemDanh)
        self.btn_DongCam_3.clicked.connect(self.DongCam)
        self.btn_BackPageTKB_3.clicked.connect(self.BackLichDay)
        self.btn_SaveDD_3.clicked.connect(self.SaveDD)
        ######################################################################################################
        # page giảng viên
        self.loadPageGV()
        self.themgv=False
        self.tbl_DSGV_7.clicked.connect(self.GetGV_7)
        self.btn_ThemGV_7.clicked.connect(self.ThemGV)
        self.btn_SuaGV_7.clicked.connect(self.SuaGV)
        self.btn_HuyGV_7.clicked.connect(self.HuyGV)
        self.btn_TimGV_7.clicked.connect(self.TimGV)
        self.btn_ChonAnhGV_7.clicked.connect(self.ChonAnhGV)
        self.btn_LuuGV_7.clicked.connect(self.LuuGV)
        self.btn_XoaGV_7.clicked.connect(self.XoaGV)
        ######################################################################################################
        # page chi tiết điểm danh
        self.tbl_DSBuoiHoc_8.clicked.connect(self.GetDD)
        self.btn_BackPageBuoiHoc.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_LichHoc))

    ###chung
    def GetThoiGian(self):
        string = time.strftime('%H:%M:%S %p')
        self.lbl_time.setText(string)
    def chuyenGio(self,str):
        h=int(str.split(":")[0])
        m=str.split(":")[1].split(" ")[0]
        s=str.split(":")[1].split(" ")[1]
        if s=="CH":
            h+=12
        kq = f"{h}:{m} "
        return kq
    ######################################################################################################
    # page chi tiết điểm danh
    def GetDD(self):
        row = self.tbl_DSBuoiHoc_8.currentIndex().row()
        ms = self.tbl_DSBuoiHoc_8.item(row, 0).text()
        lstBH = []
        for row1 in cursor.execute(f"select * from CHITIETDD,SINHVIEN WHERE CHITIETDD.MASV=SINHVIEN.MASV AND MABH='{ms}'"):
            LHP = {}
            LHP["ID"] = row1[6]
            LHP["name"] = row1[7]
            LHP["tt"] = row1[5]
            LHP["time"] = row1[3]
            LHP["ghichu"] = row1[4]
            LHP["hinhanh"] = row1[2]
            lstBH.append(LHP)
        self.tbl_ChiTietDD_8.setColumnWidth(0, 130)
        self.tbl_ChiTietDD_8.setColumnWidth(1, 200)
        self.tbl_ChiTietDD_8.setColumnWidth(2, 150)
        self.tbl_ChiTietDD_8.setColumnWidth(3, 120)
        self.tbl_ChiTietDD_8.setColumnWidth(4, 150)
        self.tbl_ChiTietDD_8.setColumnWidth(5, 170)
        self.tbl_ChiTietDD_8.verticalHeader().setDefaultSectionSize(150)
        self.tbl_ChiTietDD_8.setRowCount(len(lstBH))
        row1=0
        for mh in lstBH:
            self.tbl_ChiTietDD_8.setItem(row1, 0, QtWidgets.QTableWidgetItem(mh["ID"]))
            self.tbl_ChiTietDD_8.setItem(row1, 1, QtWidgets.QTableWidgetItem(mh["name"]))
            self.tbl_ChiTietDD_8.setItem(row1, 2, QtWidgets.QTableWidgetItem(mh["tt"]))
            self.tbl_ChiTietDD_8.setItem(row1, 3, QtWidgets.QTableWidgetItem(mh["time"]))
            self.tbl_ChiTietDD_8.setItem(row1, 4, QtWidgets.QTableWidgetItem(mh["ghichu"]))
            if mh["hinhanh"]!=None:
                image = np.array(Image.open(f"AnhDiemDanh/{mh['hinhanh']}"))
                img = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0],
                                   QtGui.QImage.Format_RGB888)
                imglabel = QtWidgets.QLabel("Image")
                imglabel.setPixmap(QtGui.QPixmap.fromImage(img))
                self.tbl_ChiTietDD_8.setCellWidget(row1,5,imglabel)
            else:
                pass
            row1 += 1


    ######################################################################################################
    # page giảng viên
    def XoaGV(self):
        if self.fi_IDGV_7.text()=="":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Thông báo")
            msg.setText("Vui lòng chọn giảng viên muốn xóa")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        else:
            try:
                cursor.execute("DELETE GIANGVIEN "
                               "WHERE MAGV=?", self.fi_IDGV_7.text())
                con.commit()
                msg = QMessageBox()
                msg.setIcon(QMessageBox.NoIcon)
                msg.setWindowTitle("Thông báo")
                msg.setText("Xóa thành công")
                msg.setStandardButtons(QMessageBox.Ok)
                result = msg.exec_()
                self.loadPageGV()
            except:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.NoIcon)
                msg.setWindowTitle("Thông báo")
                msg.setText("Lỗi")
                msg.setStandardButtons(QMessageBox.Ok)
                result = msg.exec_()
    def LuuGV(self):
        if self.fi_HoTenGV_7.text()=="" or self.fi_SDT_7.text()=="" or self.fi_Khoa_7.currentText()==-1 \
                or self.fi_DDThuongTru_7.text()=="" or self.fi_DCTamTru_7.text()=="" or self.fi_GioiTinhGV_7.currentText()==-1 \
                or self.fi_CMND_7.text()=="" or self.fi_DanToc_7.text()=="" or self.fi_TrinhDo_7.text()=="" \
                or self.fi_Email_7.text()=="":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Thông báo")
            msg.setText("Vui lòng nhập đầy đủ thông tin")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        else:
            if self.themgv==True:
                try:
                    hinhanh=self.fi_IDGV_7.text()+".png"
                    mk=(self.fi_NgaySinhGV_7.text().split("/")[0]+self.fi_NgaySinhGV_7.text().split("/")[1]+self.fi_NgaySinhGV_7.text().split("/")[2])
                    cursor.execute("SET DATEFORMAT DMY INSERT INTO GIANGVIEN "
                                   "Values(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", self.fi_IDGV_7.text(), self.fi_HoTenGV_7.text(),
                                   self.fi_SDT_7.text(), self.fi_NgaySinhGV_7.text(), self.fi_CMND_7.text(),self.fi_DanToc_7.text(),
                                   self.fi_TrinhDo_7.text(),
                                   self.fi_GioiTinhGV_7.currentText(),self.fi_Email_7.text(),mk,self.fi_DDThuongTru_7.text(),
                                   self.fi_DCTamTru_7.text(),hinhanh,self.fi_Khoa_7.currentText().split('-')[1])
                    cv2.imwrite("Avatar/"+f"{self.fi_IDGV_7.text()}.png",self.image_GV)
                    con.commit()
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.NoIcon)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Thêm giảng viên thành công")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()
                    self.loadPageGV()
                except:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Lỗi")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()
            elif self.themgv == False:
                try:
                    hinhanh=self.fi_IDGV_7.text()+".png"
                    cursor.execute("SET DATEFORMAT DMY UPDATE GIANGVIEN "
                                   "SET TENGV=?, SDT=?, NGAYSINH=?, CMND=?, DANTOC=?,TRINHDO=?,GIOITINH=?, EMAIL=?,DIACHITT=?, DCTAMTRU=?, MAKHOA=? WHERE MAGV=?", self.fi_HoTenGV_7.text(),
                                   self.fi_SDT_7.text(), self.fi_NgaySinhGV_7.text(), self.fi_CMND_7.text(),self.fi_DanToc_7.text(),
                                   self.fi_TrinhDo_7.text(),
                                   self.fi_GioiTinhGV_7.currentText(),self.fi_Email_7.text(),self.fi_DDThuongTru_7.text(),
                                   self.fi_DCTamTru_7.text(),self.fi_Khoa_7.currentText().split('-')[1],self.fi_IDGV_7.text())
                    cv2.imwrite("Avatar/"+f"{self.fi_IDGV_7.text()}.png",self.image_GV)
                    con.commit()
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.NoIcon)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Cập nhật thông tin thành công")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()
                    self.loadPageGV()
                except:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Lỗi")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()

    def ChonAnhGV(self):
        folder_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file','c:\'', "Image files (*.jpg *.png)")
        self.image_GV = np.array(Image.open(folder_path[0]))
        self.image_GV=cv2.cvtColor(self.image_GV,cv2.COLOR_BGR2RGB)
        image= cv2.resize(self.image_GV, (157, 115),3)
        img = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0],
                           QtGui.QImage.Format_RGB888)  # tổng số pixel quét của ảnh
        self.lbl_AvatarGV_7.setPixmap(QtGui.QPixmap.fromImage(img))
    def TimGV(self):
        lstTimGV = []
        row = 0
        if self.txt_TimGV_7.text() == "":
            for row1 in cursor.execute(
                    f"select * from GIANGVIEN"):
                SV = {}
                SV["ID"] = row1[0]
                SV["name"] = row1[1]
                lstTimGV.append(SV)

            self.tbl_DSGV_7.setColumnWidth(0, 200)
            self.tbl_DSGV_7.setColumnWidth(1, 250)
            self.tbl_DSGV_7.setRowCount(len(lstTimGV))
            for sv in lstTimGV:
                self.tbl_DSGV_7.setItem(row, 0, QtWidgets.QTableWidgetItem(sv["ID"]))
                self.tbl_DSGV_7.setItem(row, 1, QtWidgets.QTableWidgetItem(sv["name"]))
                row += 1
        elif self.cbo_TimKiemGV_7.currentText() == "ID":
            for row1 in cursor.execute(f"select * from GIANGVIEN WHERE GIANGVIEN.MAGV='{self.txt_TimGV_7.text()}'"):
                SV = {}
                SV["ID"] = row1[0]
                SV["name"] = row1[1]
                lstTimGV.append(SV)

            self.tbl_DSGV_7.setColumnWidth(0, 200)
            self.tbl_DSGV_7.setColumnWidth(1, 250)
            self.tbl_DSGV_7.setRowCount(len(lstTimGV))
            for sv in lstTimGV:
                self.tbl_DSGV_7.setItem(row, 0, QtWidgets.QTableWidgetItem(sv["ID"]))
                self.tbl_DSGV_7.setItem(row, 1, QtWidgets.QTableWidgetItem(sv["name"]))
                row += 1


        elif self.cbo_TimKiemGV_7.currentText() == "Họ Tên":
            for row1 in cursor.execute(f"select * from GIANGVIEN WHERE  GIANGVIEN.TENGV LIKE N'%{self.txt_TimGV_7.text()}%'"):
                SV = {}
                SV["ID"] = row1[0]
                SV["name"] = row1[1]
                lstTimGV.append(SV)

            self.tbl_DSGV_7.setColumnWidth(0, 200)
            self.tbl_DSGV_7.setColumnWidth(1, 250)
            self.tbl_DSGV_7.setRowCount(len(lstTimGV))
            for sv in lstTimGV:
                self.tbl_DSGV_7.setItem(row, 0, QtWidgets.QTableWidgetItem(sv["ID"]))
                self.tbl_DSGV_7.setItem(row, 1, QtWidgets.QTableWidgetItem(sv["name"]))
                row += 1

    def HuyGV(self):
        self.KhoiTaoPageGV()
        self.fi_IDGV_7.setText("")
        self.fi_HoTenGV_7.setText("")
        self.fi_SDT_7.setText("")
        self.fi_DCTamTru_7.setText("")
        self.fi_DDThuongTru_7.setText("")
        self.fi_CMND_7.setText("")
        self.fi_DanToc_7.setText("")
        self.fi_TrinhDo_7.setText("")
        self.fi_Email_7.setText("")

    def SuaGV(self):
        if self.fi_IDGV_7.text()=="":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.NoIcon)
            msg.setWindowTitle("Thông báo")
            msg.setText("Vui lòng chọn giáo viên cần cập nhật thông tin!!")
            msg.setStandardButtons(QMessageBox.Ok)
            result = msg.exec_()
            self.loadPageGV()
        else:
            self.themgv = False
            self.btn_ThemGV_7.setVisible(False)
            self.btn_SuaGV_7.setVisible(False)
            self.btn_XoaGV_7.setVisible(False)
            self.btn_LuuGV_7.setVisible(True)
            self.btn_HuyGV_7.setVisible(True)
            self.fi_HoTenGV_7.setEnabled(True)
            self.fi_SDT_7.setEnabled(True)
            self.fi_Khoa_7.setEnabled(True)
            self.fi_DCTamTru_7.setEnabled(True)
            self.fi_DDThuongTru_7.setEnabled(True)
            self.fi_GioiTinhGV_7.setEnabled(True)
            self.fi_CMND_7.setEnabled(True)
            self.fi_DanToc_7.setEnabled(True)
            self.fi_TrinhDo_7.setEnabled(True)
            self.fi_Email_7.setEnabled(True)
            self.btn_ChonAnhGV_7.setVisible(True)
    def ThemGV(self):
        self.themgv=True
        self.btn_ThemGV_7.setVisible(False)
        self.btn_SuaGV_7.setVisible(False)
        self.btn_XoaGV_7.setVisible(False)
        self.btn_LuuGV_7.setVisible(True)
        self.btn_HuyGV_7.setVisible(True)
        self.fi_HoTenGV_7.setEnabled(True)
        self.fi_SDT_7.setEnabled(True)
        self.fi_Khoa_7.setEnabled(True)
        self.fi_DCTamTru_7.setEnabled(True)
        self.fi_DDThuongTru_7.setEnabled(True)
        self.fi_GioiTinhGV_7.setEnabled(True)
        self.fi_CMND_7.setEnabled(True)
        self.fi_DanToc_7.setEnabled(True)
        self.fi_TrinhDo_7.setEnabled(True)
        self.fi_Email_7.setEnabled(True)
        self.fi_Khoa_7.setCurrentIndex(-1)
        self.fi_GioiTinhGV_7.setCurrentIndex(-1)
        self.btn_ChonAnhGV_7.setVisible(True)
        self.fi_NgaySinhGV_7.setDate(QtCore.QDate(2000, 1, 1))
        sql = " EXEC GIANGVIEN_ID "
        cursor.execute(sql)
        data = cursor.fetchall()
        self.fi_IDGV_7.setText(data[0][0])
        self.fi_HoTenGV_7.setText("")
        self.fi_SDT_7.setText("")
        self.fi_DCTamTru_7.setText("")
        self.fi_DDThuongTru_7.setText("")
        self.fi_CMND_7.setText("")
        self.fi_DanToc_7.setText("")
        self.fi_TrinhDo_7.setText("")
        self.fi_Email_7.setText("")
        image = np.array(Image.open(f"Avatar/Who1.png"))
        image = cv2.resize(image, (157, 150))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0],
                           QtGui.QImage.Format_RGB888)  # tổng số pixel quét của ảnh
        self.lbl_AvatarGV_7.setPixmap(QtGui.QPixmap.fromImage(img))
    def GetGV_7(self):
        row = self.tbl_DSGV_7.currentIndex().row()
        ms = self.tbl_DSGV_7.item(row, 0).text()
        for row in cursor.execute(f"select * from GIANGVIEN,KHOA WHERE MAGV='{ms}' AND KHOA.MAKHOA=GIANGVIEN.MAKHOA"):
            self.fi_IDGV_7.setText(row[0])
            self.fi_HoTenGV_7.setText(row[1])
            self.fi_SDT_7.setText(row[2])
            self.fi_NgaySinhGV_7.setDate(row[3])
            self.fi_CMND_7.setText(row[4])
            self.fi_DanToc_7.setText(row[5])
            self.fi_TrinhDo_7.setText(row[6])
            self.fi_GioiTinhGV_7.setCurrentText(row[7])
            self.fi_Email_7.setText(row[8])
            self.fi_DDThuongTru_7.setText(row[10])
            self.fi_DCTamTru_7.setText(row[11])
            self.fi_Khoa_7.setCurrentText(row[15]+"-"+row[14])
            if row[12] == None:
                image = np.array(Image.open(f"Avatar/Who1.png"))
                self.image_GV=image
                image = cv2.resize(image, (157, 150))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0],
                                   QtGui.QImage.Format_RGB888)  # tổng số pixel quét của ảnh
                self.lbl_AvatarGV_7.setPixmap(QtGui.QPixmap.fromImage(img))
            else:
                image = np.array(Image.open(f"Avatar/{row[12]}"))
                self.image_GV = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (157, 150))
                img = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0],
                                   QtGui.QImage.Format_RGB888)  # tổng số pixel quét của ảnh
                self.lbl_AvatarGV_7.setPixmap(QtGui.QPixmap.fromImage(img))
    def loadPageGV(self):
        #load table giảng viên
        row = 0
        lstGV = []
        for row1 in cursor.execute(f"select * from GIANGVIEN"):
            LHP = {}
            LHP["ID"] = row1[0]
            LHP["name"] = row1[1]
            lstGV.append(LHP)

        self.tbl_DSGV_7.setColumnWidth(0, 200)
        self.tbl_DSGV_7.setColumnWidth(1, 250)
        self.tbl_DSGV_7.setRowCount(len(lstGV))
        for mh in lstGV:
            self.tbl_DSGV_7.setItem(row, 0, QtWidgets.QTableWidgetItem(mh["ID"]))
            self.tbl_DSGV_7.setItem(row, 1, QtWidgets.QTableWidgetItem(mh["name"]))
            row += 1
        #load combobox tìm kiếm giảng viên
        self.cbo_TimKiemGV_7.clear()
        self.cbo_TimKiemGV_7.addItem("ID")
        self.cbo_TimKiemGV_7.addItem("Họ Tên")
        # load combobox khoa
        self.fi_Khoa_7.clear()
        for row1 in cursor.execute(f"select * from KHOA"):
            self.fi_Khoa_7.addItem(row1[1]+"-"+row1[0])

        self.fi_GioiTinhGV_7.clear()

        self.fi_GioiTinhGV_7.addItem("Nam")
        self.fi_GioiTinhGV_7.addItem("Nữ")
        self.fi_GioiTinhGV_7.addItem("Khác")
        self.KhoiTaoPageGV()

    def KhoiTaoPageGV(self):
        self.btn_ThemGV_7.setVisible(True)
        self.btn_SuaGV_7.setVisible(True)
        self.btn_XoaGV_7.setVisible(True)
        self.btn_LuuGV_7.setVisible(False)
        self.btn_HuyGV_7.setVisible(False)
        self.fi_IDGV_7.setEnabled(False)
        self.fi_HoTenGV_7.setEnabled(False)
        self.fi_SDT_7.setEnabled(False)
        self.fi_Khoa_7.setEnabled(False)
        self.fi_DCTamTru_7.setEnabled(False)
        self.fi_DDThuongTru_7.setEnabled(False)
        self.fi_GioiTinhGV_7.setEnabled(False)
        self.fi_CMND_7.setEnabled(False)
        self.fi_DanToc_7.setEnabled(False)
        self.fi_TrinhDo_7.setEnabled(False)
        self.fi_Email_7.setEnabled(False)
        self.fi_Khoa_7.setCurrentIndex(-1)
        self.fi_GioiTinhGV_7.setCurrentIndex(-1)
        self.btn_ChonAnhGV_7.setVisible(False)
        self.fi_NgaySinhGV_7.setDate(QtCore.QDate(2000,1,1))
        image = np.array(Image.open(f"Avatar/Who1.png"))
        image = cv2.resize(image, (157, 150))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0],
                           QtGui.QImage.Format_RGB888)  # tổng số pixel quét của ảnh
        self.lbl_AvatarGV_7.setPixmap(QtGui.QPixmap.fromImage(img))

    ######################################################################################################
    # page lịch dạy
    def GetBuoiHocDD(self):
        row = self.tbl_DSBuoiHoc_6.currentIndex().row()
        ms = self.tbl_DSBuoiHoc_6.item(row, 0).text()
        for row in cursor.execute(f"select * from BUOIHOC WHERE MABH='{ms}'"):
            self.fi_IDBH_6.setText(row[0])
            self.fi_Ngay_6.setDate(row[2])
            self.fi_GioBD_6.setTime(QtCore.QTime(int(row[3].split(":")[0]), int(row[3].split(":")[1])))
            self.fi_GioKT_6.setTime(QtCore.QTime(int(row[4].split(":")[0]), int(row[4].split(":")[1])))
            self.fi_Phong_6.setText(row[5])

    def GetLopHPDD(self):
        row = self.tbl_DSLopHP_6.currentIndex().row()
        ms = self.tbl_DSLopHP_6.item(row, 0).text()
        for row in cursor.execute(f"select * from LOPMONHOC,MONHOC, GIANGVIEN WHERE LOPMONHOC.MALOPMH='{ms}' "
                                  f"AND LOPMONHOC.MAMH=MONHOC.MAMH AND LOPMONHOC.MAGV=GIANGVIEN.MAGV "):
            self.fi_MonHoc_6.setText(row[7])
            self.fi_IDLop_6.setText(row[0])
            self.fi_GiaoVien_6.setText(row[13])
            self.fi_SiSo_6.setText(str(row[5]))
            self.fi_NgayBD_6.setDate(row[2])
            self.fi_NgayKT_6.setDate(row[3])

        row = 0
        lstBuoiHoc = []
        for row1 in cursor.execute(f"select * from BUOIHOC WHERE MALOPMH='{ms}'"):
            LHP = {}
            LHP["ID"] = row1[0]
            y = str(row1[2]).split("-")[0]
            m = str(row1[2]).split("-")[1]
            d = str(row1[2]).split("-")[2]
            LHP["date"] = d + "/" + m + "/" + y
            lstBuoiHoc.append(LHP)

        self.tbl_DSBuoiHoc_6.setColumnWidth(0, 150)
        self.tbl_DSBuoiHoc_6.setColumnWidth(1, 150)
        self.tbl_DSBuoiHoc_6.setRowCount(len(lstBuoiHoc))
        for mh in lstBuoiHoc:
            self.tbl_DSBuoiHoc_6.setItem(row, 0, QtWidgets.QTableWidgetItem(mh["ID"]))
            self.tbl_DSBuoiHoc_6.setItem(row, 1, QtWidgets.QTableWidgetItem(mh["date"]))
            row += 1

    def loadThoiKhoaBieu(self):
        row = 0
        self.lstTKB = []
        for row1 in cursor.execute(f"select LOPMONHOC.MALOPMH ,MONHOC.TENMH from LOPMONHOC, MONHOC WHERE MONHOC.MAMH=LOPMONHOC.MAMH"):
            LHP = {}
            LHP["ID"] = row1[0]
            LHP["MH"] = row1[1]
            self.lstTKB.append(LHP)
        self.tbl_DSLopHP_6.setColumnWidth(0, 150)
        self.tbl_DSLopHP_6.setColumnWidth(1, 250)
        self.tbl_DSLopHP_6.setRowCount(len(self.lstTKB))
        for mh in self.lstTKB:
            self.tbl_DSLopHP_6.setItem(row, 0, QtWidgets.QTableWidgetItem(mh["ID"]))
            self.tbl_DSLopHP_6.setItem(row, 1, QtWidgets.QTableWidgetItem(mh["MH"]))
            row += 1
        self.fi_MonHoc_6.setEnabled(False)
        self.fi_GiaoVien_6.setEnabled(False)
        self.fi_NgayBD_6.setEnabled(False)
        self.fi_IDLop_6.setEnabled(False)
        self.fi_SiSo_6.setEnabled(False)
        self.fi_NgayKT_6.setEnabled(False)
        self.fi_IDBH_6.setEnabled(False)
        self.fi_Ngay_6.setEnabled(False)
        self.fi_GioBD_6.setEnabled(False)
        self.fi_GioKT_6.setEnabled(False)
        self.fi_Phong_6.setEnabled(False)

    def OpenDiemDanh(self):
        if self.fi_IDBH_6.text()=="":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Thông báo")
            msg.setText("Vui lòng chọn buổi học ")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        else:
            self.fi_MonHoc_33.setText(self.fi_MonHoc_6.text())
            self.fi_IDBH_33.setText(self.fi_IDBH_6.text())
            self.fi_SiSo_3.setText(self.fi_SiSo_6.text())
            self.fi_LopHoc_3.setText(self.fi_IDLop_6.text())
            self.fi_GiangVien_3.setText(self.fi_GiaoVien_6.text())
            self.fi_Phong_3.setText(self.fi_Phong_6.text())
            h = int(self.fi_GioBD_6.text().split(":")[0])
            m = int(self.fi_GioBD_6.text().split(":")[1].split(" ")[0])
            self.fi_GioBD_3.setTime(QtCore.QTime(h,m))
            h1 = int(self.fi_GioKT_6.text().split(":")[0])
            m1 = int(self.fi_GioKT_6.text().split(":")[1].split(" ")[0])
            self.fi_GioKT_3.setTime(QtCore.QTime(h1, m1))
            self.stackedWidget.setCurrentWidget(self.page_DiemDanh)
            self.fi_MonHoc_33.setEnabled(False)
            self.fi_IDBH_33.setEnabled(False)
            self.fi_SiSo_3.setEnabled(False)
            self.fi_LopHoc_3.setEnabled(False)
            self.fi_GiangVien_3.setEnabled(False)
            self.fi_Phong_3.setEnabled(False)
            self.fi_GioKT_3.setEnabled(False)
            self.fi_GioBD_3.setEnabled(False)
            self.fi_IDSV_3.setEnabled(False)
            self.fi_HoTenSV_3.setEnabled(False)
            self.fi_ThoiGianDD_3.setEnabled(False)
            self.fi_IDSV_3.setText("")
            self.fi_HoTenSV_3.setText("")
            self.fi_ThoiGianDD_3.setTime(QtCore.QTime(12,0))
            #load sinh viên trong lớp
            row = 0
            lstSVBH = []
            for row1 in cursor.execute(f"select * from CHITIETLHP,SINHVIEN WHERE SINHVIEN.MASV=CHITIETLHP.MASV AND MALOPMH='{self.fi_IDLop_6.text()}'"):
                LHP = {}
                LHP["ID"] = row1[1]
                LHP["name"]=row1[3]
                LHP["dd"] = "vắng"
                LHP["ghichu"] = ""
                LHP["tg"] = ""
                lstSVBH.append(LHP)
            self.tbl_DSSV_3.setColumnWidth(0, 120)
            self.tbl_DSSV_3.setColumnWidth(1, 180)
            self.tbl_DSSV_3.setColumnWidth(2, 120)
            self.tbl_DSSV_3.setColumnWidth(3, 120)
            self.tbl_DSSV_3.setColumnWidth(4, 120)
            self.tbl_DSSV_3.setRowCount(len(lstSVBH))
            for mh in lstSVBH:
                self.tbl_DSSV_3.setItem(row, 0, QtWidgets.QTableWidgetItem(mh["ID"]))
                self.tbl_DSSV_3.setItem(row, 1, QtWidgets.QTableWidgetItem(mh["name"]))
                self.tbl_DSSV_3.setItem(row, 2, QtWidgets.QTableWidgetItem(mh["dd"]))
                self.tbl_DSSV_3.setItem(row, 3, QtWidgets.QTableWidgetItem(mh["tg"]))
                self.tbl_DSSV_3.setItem(row, 4, QtWidgets.QTableWidgetItem(mh["ghichu"]))

                row += 1
    ######################################################################################################
    # page điểm danh
    def SaveDD(self):
        try:
            for i in range(self.tbl_DSSV_3.rowCount()):
                hinhanh=self.fi_IDBH_33.text()+"_"+self.tbl_DSSV_3.item(i, 0).text()+".png"
                if self.tbl_DSSV_3.item(i, 2).text() == "có mặt":
                    cursor.execute("INSERT INTO CHITIETDD "
                                       "Values(?,?,?,?,?,?)", self.fi_IDBH_33.text()
                                   , self.tbl_DSSV_3.item(i, 0).text(),hinhanh,self.tbl_DSSV_3.item(i, 3).text(),
                                   self.tbl_DSSV_3.item(i, 4).text(), self.tbl_DSSV_3.item(i, 2).text())
                    con.commit()
                elif self.tbl_DSSV_3.item(i, 2).text() == "vắng":
                    cursor.execute("INSERT INTO CHITIETDD(MABH,MASV,GHICHU, TINHTRANG) "
                                   "Values(?,?,?,?)", self.fi_IDBH_33.text()
                                   , self.tbl_DSSV_3.item(i, 0).text(), self.tbl_DSSV_3.item(i, 4).text(),
                                   self.tbl_DSSV_3.item(i, 2).text())
                    con.commit()
            msg = QMessageBox()
            msg.setIcon(QMessageBox.NoIcon)
            msg.setWindowTitle("Thông báo")
            msg.setText("Thành công")
            msg.setStandardButtons(QMessageBox.Ok)
            result = msg.exec_()
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Thông báo")
            msg.setText("Lỗi")
            msg.setStandardButtons(QMessageBox.Ok)
            result = msg.exec_()

    def BackLichDay(self):
        self.stackedWidget.setCurrentWidget(self.page_LichDay)
        self.DongCam()
    def DongCam(self):
        self.t = 0
        image = cv2.imread("Avatar/pic.png")
        image = cv2.resize(image, (721, 401))
        img = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0],
                               QtGui.QImage.Format_RGB888)
        self.lbl_Cam_3.setPixmap(QtGui.QPixmap.fromImage(img))
        image = np.array(Image.open(f"Avatar/Who1.png"))
        image = cv2.resize(image, (173, 120))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0],
                           QtGui.QImage.Format_RGB888)  # tổng số pixel quét của ảnh
        self.lbl_AvatarDD_3.setPixmap(QtGui.QPixmap.fromImage(img))
        self.fi_IDSV_3.setText("")
        self.fi_HoTenSV_3.setText("")
        self.fi_ThoiGianDD_3.setTime(QtCore.QTime(12,0))
    def ThucHienDD(self):
        if self.fi_IDSV_3.text()!="":
            for row in cursor.execute(f"select * from SINHVIEN Where MASV='{self.fi_IDSV_3.text()}'"):
                self.fi_HoTenSV_3.setText(row[1])
                if row[11] == None:
                    image = np.array(Image.open(f"Avatar/Who1.png"))
                    image = cv2.resize(image, (205, 109))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    img = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0],
                                       QtGui.QImage.Format_RGB888)  # tổng số pixel quét của ảnh
                    self.lbl_AvatarDD_3.setPixmap(QtGui.QPixmap.fromImage(img))
                else:
                    image = np.array(Image.open(f"Avatar/{row[11]}"))
                    image = cv2.resize(image, (173, 120))

                    img = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0],
                                       QtGui.QImage.Format_RGB888)  # tổng số pixel quét của ảnh
                    self.lbl_AvatarDD_3.setPixmap(QtGui.QPixmap.fromImage(img))
                    string = time.strftime('%H:%M:%S %p')
                    self.fi_ThoiGianDD_3.setTime(QtCore.QTime(int(string.split(":")[0]),int(string.split(":")[1])))
                    for i in range(self.tbl_DSSV_3.rowCount()):
                        if self.tbl_DSSV_3.item(i,0).text() ==self.fi_IDSV_3.text():
                            self.tbl_DSSV_3.setItem(i, 2, QtWidgets.QTableWidgetItem("có mặt"))
                            self.tbl_DSSV_3.setItem(i, 3, QtWidgets.QTableWidgetItem(self.fi_ThoiGianDD_3.text()))
                            h1=self.chuyenGio(self.fi_GioBD_3.text())
                            h2=self.chuyenGio(self.fi_ThoiGianDD_3.text())
                            if int(h1.split(":")[0])<int(h2.split(":")[0]):
                                self.tbl_DSSV_3.setItem(i, 4, QtWidgets.QTableWidgetItem("quá trễ"))
                            elif int(h1.split(":")[0])==int(h2.split(":")[0]):
                                if int(h1.split(":")[1])<int(h2.split(":")[1]):
                                    tre=int(h2.split(":")[1])-int(h1.split(":")[1])
                                    self.tbl_DSSV_3.setItem(i, 4, QtWidgets.QTableWidgetItem("trễ "+str(tre)+" phút"))


    def OpenCamDiemDanh(self):
        self.FPS = 0
        st = 0
        self.cam = cv2.VideoCapture(0)
        database = {}
        myfile = open("data.pkl", "rb")
        database = pickle.load(myfile)
        myfile.close()
        while True:
            Ok, self.frame = self.cam.read()
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            faces = face_detector.detect_faces(self.frame)  # cắt gương mặt
            for face in faces:
                bounding_box = face['box']
                try:
                    self.cuttedFace = self.frame[bounding_box[1]:bounding_box[1] + bounding_box[3],
                         bounding_box[0]:bounding_box[0] + bounding_box[2]]

                    cv2.rectangle(self.frame,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0,155,255),
                      2)
                    try:
                        self.cuttedFace = cv2.resize(self.cuttedFace, (160, 160))
                        faceDD=cv2.cvtColor(self.cuttedFace,cv2.COLOR_BGR2RGB)
                        mean, std = self.cuttedFace.mean(), self.cuttedFace.std()
                        self.cuttedFace= (self.cuttedFace - mean) / std
                        self.cuttedFace= expand_dims(self.cuttedFace, axis=0)
                        signature = MyFaceNet.predict(self.cuttedFace)
                        min_dist = 100
                        identity = ' '
                        for key, value in database.items():
                            dist = np.linalg.norm(value - signature)
                            if dist < min_dist:
                                min_dist = dist
                                identity = key
                        for i in range(self.tbl_DSSV_3.rowCount()):
                            if self.tbl_DSSV_3.item(i, 0).text() == identity:
                                self.fi_IDSV_3.setText(identity)
                                cv2.putText(self.frame, identity, (bounding_box[0], bounding_box[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
                                            cv2.LINE_AA)
                                cv2.imwrite("AnhDiemDanh/"+f"{self.fi_IDBH_33.text()}_{self.fi_IDSV_3.text()}.png",
                                            faceDD)
                                self.ThucHienDD()
                    except:
                        pass
                except:
                    pass
            self.update()
            self.t = 1
            if cv2.waitKey(1) == ord('q') or self.t == 0:
                break
        self.cam.release()
        cv2.destroyAllWindows()

    def update(self):

        text = "Time:" + str(time.strftime(("%H:%M %p")))
        ps.putBText(self.frame, text, text_offset_x=10, text_offset_y=10, vspace=10, hspace=10, font_scale=0.8,
                        background_RGB=(100, 100, 100), text_RGB=(199, 150, 255))
        self.setPhoto(self.frame)


    def setPhoto(self, image):
        image = cv2.resize(image, (721, 401))
        img = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0],
                               QtGui.QImage.Format_RGB888)  # tổng số pixel quét của ảnh
        self.lbl_Cam_3.setPixmap(QtGui.QPixmap.fromImage(img))
    ######################################################################################################
    # page buổi học
    def ChiTietDD(self):
        self.stackedWidget.setCurrentWidget(self.page_ChiTietDD)
        row = 0
        lstBH = []
        for row1 in cursor.execute(f"select * from BUOIHOC WHERE MALOPMH='{self.fi_IDLHP_4.text()}'"):
            LHP = {}
            LHP["ID"] = row1[0]
            y = str(row1[2]).split("-")[0]
            m = str(row1[2]).split("-")[1]
            d = str(row1[2]).split("-")[2]
            LHP["date"] = d + "/" + m + "/" + y
            lstBH.append(LHP)
        self.tbl_DSBuoiHoc_8.setColumnWidth(0, 150)
        self.tbl_DSBuoiHoc_8.setColumnWidth(1, 150)
        self.tbl_DSBuoiHoc_8.setRowCount(len(lstBH))
        for mh in lstBH:
            self.tbl_DSBuoiHoc_8.setItem(row, 0, QtWidgets.QTableWidgetItem(mh["ID"]))
            self.tbl_DSBuoiHoc_8.setItem(row, 1, QtWidgets.QTableWidgetItem(mh["date"]))
            row += 1
    def XoaBH(self):
        if self.fi_IDBH_4.text()=="":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Thông báo")
            msg.setText("Vui lòng buổi học muốn xóa")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        else:
            try:
                cursor.execute("DELETE BUOIHOC "
                               "WHERE MABH=?", self.fi_IDBH_4.text())
                con.commit()
                msg = QMessageBox()
                msg.setIcon(QMessageBox.NoIcon)
                msg.setWindowTitle("Thông báo")
                msg.setText("Xóa thành công")
                msg.setStandardButtons(QMessageBox.Ok)
                result = msg.exec_()
                self.KhoiTaoBH()
                row = 0
                self.lstBH = []
                for row1 in cursor.execute(f"select * from BUOIHOC WHERE MALOPMH='{self.fi_IDLHP_4.text()}'"):
                    LHP = {}
                    LHP["ID"] = row1[0]
                    y = str(row1[2]).split("-")[0]
                    m = str(row1[2]).split("-")[1]
                    d = str(row1[2]).split("-")[2]
                    LHP["date"] = d + "/" + m + "/" + y
                    LHP["room"] = row1[5]
                    self.lstBH.append(LHP)
                self.tbl_DSBuoiHoc_4.setColumnWidth(0, 150)
                self.tbl_DSBuoiHoc_4.setColumnWidth(1, 150)
                self.tbl_DSBuoiHoc_4.setColumnWidth(2, 110)
                self.tbl_DSBuoiHoc_4.setRowCount(len(self.lstBH))
                for mh in self.lstBH:
                    self.tbl_DSBuoiHoc_4.setItem(row, 0, QtWidgets.QTableWidgetItem(mh["ID"]))
                    self.tbl_DSBuoiHoc_4.setItem(row, 1, QtWidgets.QTableWidgetItem(mh["date"]))
                    self.tbl_DSBuoiHoc_4.setItem(row, 2, QtWidgets.QTableWidgetItem(mh["room"]))
                    row += 1
            except:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Thông báo")
                msg.setText("Lỗi")
                msg.setStandardButtons(QMessageBox.Ok)
                result = msg.exec_()
    def LuuBH(self):
        if self.thembh == True:
            if self.fi_PhongHoc_4.currentIndex == -1:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Thông báo")
                msg.setText("Vui lòng nhập đầy đủ thông tin!!")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            else:
                try:
                    bd=self.chuyenGio(self.fi_GioBD_4.text())
                    kt = self.chuyenGio(self.fi_GioKT_4.text())
                    cursor.execute("SET DATEFORMAT DMY INSERT INTO BUOIHOC "
                                       "Values(?,?,?,?,?,?)", self.fi_IDBH_4.text(), self.fi_IDLHP_4.text(),
                                       self.fi_NgayHoc_4.text(),
                                       bd, kt,self.fi_PhongHoc_4.currentText())
                    con.commit()
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.NoIcon)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Thêm buổi học thành công")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()
                    self.KhoiTaoBH()
                    row = 0
                    self.lstBH = []
                    for row1 in cursor.execute(f"select * from BUOIHOC WHERE MALOPMH='{self.fi_IDLHP_4.text()}'"):
                        LHP = {}
                        LHP["ID"] = row1[0]
                        y = str(row1[2]).split("-")[0]
                        m = str(row1[2]).split("-")[1]
                        d = str(row1[2]).split("-")[2]
                        LHP["date"] = d + "/" + m + "/" + y
                        LHP["room"] = row1[5]
                        self.lstBH.append(LHP)
                    self.tbl_DSBuoiHoc_4.setColumnWidth(0, 150)
                    self.tbl_DSBuoiHoc_4.setColumnWidth(1, 150)
                    self.tbl_DSBuoiHoc_4.setColumnWidth(2, 110)
                    self.tbl_DSBuoiHoc_4.setRowCount(len(self.lstBH))
                    for mh in self.lstBH:
                        self.tbl_DSBuoiHoc_4.setItem(row, 0, QtWidgets.QTableWidgetItem(mh["ID"]))
                        self.tbl_DSBuoiHoc_4.setItem(row, 1, QtWidgets.QTableWidgetItem(mh["date"]))
                        self.tbl_DSBuoiHoc_4.setItem(row, 2, QtWidgets.QTableWidgetItem(mh["room"]))
                        row += 1
                except:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Lỗi")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()
        elif self.thembh == False:
            try:
                bd = self.chuyenGio(self.fi_GioBD_4.text())
                kt = self.chuyenGio(self.fi_GioKT_4.text())
                cursor.execute("SET DATEFORMAT DMY UPDATE BUOIHOC "
                                       "SET NGAYHOC=?,GIOBD=?,GIOKT=?, MAPH=? WHERE MABH=?",self.fi_NgayHoc_4.text(),
                                       bd, kt, self.fi_PhongHoc_4.currentText(), self.fi_IDBH_4.text())
                con.commit()
                msg = QMessageBox()
                msg.setIcon(QMessageBox.NoIcon)
                msg.setWindowTitle("Thông báo")
                msg.setText("Cập nhật buổi học thành công")
                msg.setStandardButtons(QMessageBox.Ok)
                result = msg.exec_()
                self.KhoiTaoBH()
                row = 0
                self.lstBH = []
                for row1 in cursor.execute(f"select * from BUOIHOC WHERE MALOPMH='{self.fi_IDLHP_4.text()}'"):
                    LHP = {}
                    LHP["ID"] = row1[0]
                    y = str(row1[2]).split("-")[0]
                    m = str(row1[2]).split("-")[1]
                    d = str(row1[2]).split("-")[2]
                    LHP["date"] = d + "/" + m + "/" + y
                    LHP["room"] = row1[5]
                    self.lstBH.append(LHP)
                self.tbl_DSBuoiHoc_4.setColumnWidth(0, 150)
                self.tbl_DSBuoiHoc_4.setColumnWidth(1, 150)
                self.tbl_DSBuoiHoc_4.setColumnWidth(2, 110)
                self.tbl_DSBuoiHoc_4.setRowCount(len(self.lstBH))
                for mh in self.lstBH:
                    self.tbl_DSBuoiHoc_4.setItem(row, 0, QtWidgets.QTableWidgetItem(mh["ID"]))
                    self.tbl_DSBuoiHoc_4.setItem(row, 1, QtWidgets.QTableWidgetItem(mh["date"]))
                    self.tbl_DSBuoiHoc_4.setItem(row, 2, QtWidgets.QTableWidgetItem(mh["room"]))
                    row += 1
            except:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Thông báo")
                msg.setText("Lỗi")
                msg.setStandardButtons(QMessageBox.Ok)
                result = msg.exec_()
    def GetBH(self):
        row = self.tbl_DSBuoiHoc_4.currentIndex().row()
        ms = self.tbl_DSBuoiHoc_4.item(row, 0).text()
        for row in cursor.execute(f"select * from BUOIHOC WHERE BUOIHOC.MABH='{ms}' "):
            self.fi_IDBH_4.setText(ms)
            self.fi_NgayHoc_4.setDate(row[2])
            self.fi_PhongHoc_4.setCurrentText(row[5])
            self.fi_GioBD_4.setTime(QtCore.QTime(int(row[3].split(":")[0]),int(row[3].split(":")[1])))
            self.fi_GioKT_4.setTime(QtCore.QTime(int(row[4].split(":")[0]), int(row[4].split(":")[1])))
    def SuaBH(self):
        if self.fi_IDBH_4.text()=="":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.NoIcon)
            msg.setWindowTitle("Thông báo")
            msg.setText("Vui lòng chọn buổi học cần cập nhật thông tin!!")
            msg.setStandardButtons(QMessageBox.Ok)
            result = msg.exec_()
            self.loadPageGV()
        else:
            self.thembh = False
            self.btn_ThemBH_4.setVisible(False)
            self.btn_SuaBH_4.setVisible(False)
            self.btn_XoaBH_4.setVisible(False)
            self.btn_LuuBH_4.setVisible(True)
            self.btn_HuyBH_4.setVisible(True)
            self.fi_GioBD_4.setEnabled(True)
            self.fi_GioKT_4.setEnabled(True)
            self.fi_PhongHoc_4.setEnabled(True)
            self.fi_NgayHoc_4.setEnabled(True)
    def HuyBH(self):
        self.KhoiTaoBH()
    def ThemBH(self):
        self.thembh=True
        self.btn_ThemBH_4.setVisible(False)
        self.btn_SuaBH_4.setVisible(False)
        self.btn_XoaBH_4.setVisible(False)
        self.btn_LuuBH_4.setVisible(True)
        self.btn_HuyBH_4.setVisible(True)
        mabh=self.taoMaBH(self.fi_IDLHP_4.text())
        self.fi_IDBH_4.setText(mabh)
        self.fi_NgayHoc_4.setDate(QtCore.QDate(2000,1,1))
        self.fi_PhongHoc_4.setCurrentIndex(-1)
        self.fi_GioBD_4.setTime(QtCore.QTime(12,0))
        self.fi_GioKT_4.setTime(QtCore.QTime(12, 0))
        self.fi_GioBD_4.setEnabled(True)
        self.fi_GioKT_4.setEnabled(True)
        self.fi_PhongHoc_4.setEnabled(True)
        self.fi_NgayHoc_4.setEnabled(True)
    def KhoiTaoBH(self):
        self.btn_ThemBH_4.setVisible(True)
        self.btn_SuaBH_4.setVisible(True)
        self.btn_XoaBH_4.setVisible(True)
        self.btn_LuuBH_4.setVisible(False)
        self.btn_HuyBH_4.setVisible(False)
        self.fi_IDLHP_4.setEnabled(False)
        self.fi_IDBH_4.setEnabled(False)
        self.fi_NgayHoc_4.setEnabled(False)
        self.fi_GioBD_4.setEnabled(False)
        self.fi_MH_4.setEnabled(False)
        self.fi_GiangVien_4.setEnabled(False)
        self.fi_PhongHoc_4.setEnabled(False)
        self.fi_GioKT_4.setEnabled(False)
        self.fi_PhongHoc_4.setCurrentIndex(-1)
        self.fi_IDBH_4.setText("")
    def loadBuoiHoc(self):
        #load phòng
        self.fi_PhongHoc_4.clear()
        for row1 in cursor.execute(f"select * from PHONGHOC"):
            self.fi_PhongHoc_4.addItem(row1[0])
        #load table buổi học
        row=0
        self.lstBH=[]
        for row1 in cursor.execute(f"select * from BUOIHOC WHERE MALOPMH='{self.fi_IDLHP_4.text()}'"):
            LHP = {}
            LHP["ID"] = row1[0]
            y=str(row1[2]).split("-")[0]
            m = str(row1[2]).split("-")[1]
            d = str(row1[2]).split("-")[2]
            LHP["date"] = d+"/"+m+"/"+y
            LHP["room"]=row1[5]
            self.lstBH.append(LHP)
        self.tbl_DSBuoiHoc_4.setColumnWidth(0, 150)
        self.tbl_DSBuoiHoc_4.setColumnWidth(1, 150)
        self.tbl_DSBuoiHoc_4.setColumnWidth(2, 110)
        self.tbl_DSBuoiHoc_4.setRowCount(len(self.lstBH))
        for mh in self.lstBH:
            self.tbl_DSBuoiHoc_4.setItem(row, 0, QtWidgets.QTableWidgetItem(mh["ID"]))
            self.tbl_DSBuoiHoc_4.setItem(row, 1, QtWidgets.QTableWidgetItem(mh["date"]))
            self.tbl_DSBuoiHoc_4.setItem(row, 2, QtWidgets.QTableWidgetItem(mh["room"]))
            row += 1
    def taoMaBH(self,malhp):
        cursor.execute(f"select count(MABH)+1 from BUOIHOC WHERE MALOPMH='{malhp}'")
        count= int(cursor.fetchall()[0][0])
        ma="Buoi_"+str(count)+"_"+malhp
        b=False
        while b==False:
            cursor.execute(f"select * from BUOIHOC WHERE MABH='{ma}'")
            bh = cursor.fetchall()
            if bh==[]:
                return ma
            else:
                count+=1
                ma = "Buoi_" + str(count) + "_" + malhp






    ######################################################################################################
    # page môn học

    def DatLichHoc(self):
        if self.fi_IDLop_2.text()=="":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Thông báo")
            msg.setText("Vui lòng chọn lớp học muốn xếp lịch học!!")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        else:
            self.fi_IDLHP_4.setText(self.fi_IDLop_2.text())
            self.fi_MH_4.setText(self.fi_TenMonHoc_2.text())
            self.fi_GiangVien_4.setText(self.fi_GiangVien_2.currentText().split("-")[0])
            self.stackedWidget.setCurrentWidget(self.page_LichHoc)
            self.loadBuoiHoc()

    def AllMH(self):
        row = 0
        self.lstMH = []
        for row1 in cursor.execute(f"select * from MONHOC"):
            MH = {}
            MH["ID"] = row1[0]
            MH["name"] = row1[1]
            MH["soTC"] = row1[2]
            self.lstMH.append(MH)
        self.tbl_MonHoc_2.setColumnWidth(0, 150)
        self.tbl_MonHoc_2.setColumnWidth(1, 280)
        self.tbl_MonHoc_2.setColumnWidth(2, 225)
        self.tbl_MonHoc_2.setRowCount(len(self.lstMH))
        for mh in self.lstMH:
            self.tbl_MonHoc_2.setItem(row, 0, QtWidgets.QTableWidgetItem(mh["ID"]))
            self.tbl_MonHoc_2.setItem(row, 1, QtWidgets.QTableWidgetItem(mh["name"]))
            self.tbl_MonHoc_2.setItem(row, 2, QtWidgets.QTableWidgetItem(str(mh["soTC"])))
            row += 1
    def TimMonHoc(self):
        if self.cbo_TimMH_2.currentText()=="ID":
            row = 0
            self.lstMH = []
            for row1 in cursor.execute(f"select * from MONHOC Where MAMH='{self.txt_TimMH_2.text()}'"):
                MH = {}
                MH["ID"] = row1[0]
                MH["name"] = row1[1]
                MH["soTC"] = row1[2]
                self.lstMH.append(MH)
            self.tbl_MonHoc_2.setColumnWidth(0, 150)
            self.tbl_MonHoc_2.setColumnWidth(1, 280)
            self.tbl_MonHoc_2.setColumnWidth(2, 225)
            self.tbl_MonHoc_2.setRowCount(len(self.lstMH))
            for mh in self.lstMH:
                self.tbl_MonHoc_2.setItem(row, 0, QtWidgets.QTableWidgetItem(mh["ID"]))
                self.tbl_MonHoc_2.setItem(row, 1, QtWidgets.QTableWidgetItem(mh["name"]))
                self.tbl_MonHoc_2.setItem(row, 2, QtWidgets.QTableWidgetItem(str(mh["soTC"])))
                row += 1
        elif self.cbo_TimMH_2.currentText()=="Tên Môn Học":
            row = 0
            self.lstMH = []
            for row1 in cursor.execute(f"select * from MONHOC Where TENMH LIKE N'%{self.txt_TimMH_2.text()}%'"):
                MH = {}
                MH["ID"] = row1[0]
                MH["name"] = row1[1]
                MH["soTC"] = row1[2]
                self.lstMH.append(MH)
            self.tbl_MonHoc_2.setColumnWidth(0, 150)
            self.tbl_MonHoc_2.setColumnWidth(1, 280)
            self.tbl_MonHoc_2.setColumnWidth(2, 225)
            self.tbl_MonHoc_2.setRowCount(len(self.lstMH))
            for mh in self.lstMH:
                self.tbl_MonHoc_2.setItem(row, 0, QtWidgets.QTableWidgetItem(mh["ID"]))
                self.tbl_MonHoc_2.setItem(row, 1, QtWidgets.QTableWidgetItem(mh["name"]))
                self.tbl_MonHoc_2.setItem(row, 2, QtWidgets.QTableWidgetItem(str(mh["soTC"])))
                row += 1
    def XoaSVKhoiLop(self):
        if self.fi_IDSV_2.text()=="":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Thông báo")
            msg.setText("Vui lòng chọn học sinh xóa khỏi lớp")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif self.fi_IDLop_2.text()=="":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Thông báo")
            msg.setText("Vui lòng chọn lớp học phần")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        else:
            try:
                cursor.execute("DELETE CHITIETLHP "
                               "WHERE MALOPMH=? AND MASV=?", self.fi_IDLop_2.text(), self.fi_IDSV_2.text())
                con.commit()
                msg = QMessageBox()
                msg.setIcon(QMessageBox.NoIcon)
                msg.setWindowTitle("Thông báo")
                msg.setText("Xóa thành công")
                msg.setStandardButtons(QMessageBox.Ok)
                result = msg.exec_()
                row = 0
                self.fi_IDSV_2.setText("")
                self.fi_HoTenSV_2.setText("")
                self.fi_LopSV_2.setText("")
                self.fi_KhoaSV_2.setText("")
                self.lstSVLHP = []
                for row1 in cursor.execute(f"select * from CHITIETLHP,SINHVIEN WHERE CHITIETLHP.MASV=SINHVIEN.MASV AND CHITIETLHP.MALOPMH='{self.fi_IDLop_2.text()}'"):
                    LHP = {}
                    LHP["ID"] = row1[2]
                    LHP["name"] = row1[3]
                    self.lstSVLHP.append(LHP)
                self.tbl_DSSV_2.setColumnWidth(0, 110)
                self.tbl_DSSV_2.setColumnWidth(1, 200)
                self.tbl_DSSV_2.setRowCount(len(self.lstSVLHP))
                for mh in self.lstSVLHP:
                    self.tbl_DSSV_2.setItem(row, 0, QtWidgets.QTableWidgetItem(mh["ID"]))
                    self.tbl_DSSV_2.setItem(row, 1, QtWidgets.QTableWidgetItem(mh["name"]))
                    row += 1
            except:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Thông báo")
                msg.setText("Lỗi")
                msg.setStandardButtons(QMessageBox.Ok)
                result = msg.exec_()
    def ThemSVVaoLop(self):
        if self.fi_IDSV_2.text()=="":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Thông báo")
            msg.setText("Vui lòng chọn học sinh muốn thêm vào lớp")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif self.fi_IDLop_2.text()=="":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Thông báo")
            msg.setText("Vui lòng chọn lớp học phần")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        else:
            try:
                cursor.execute("INSERT INTO CHITIETLHP "
                               "Values(?,?)", self.fi_IDLop_2.text(), self.fi_IDSV_2.text())
                con.commit()
                msg = QMessageBox()
                msg.setIcon(QMessageBox.NoIcon)
                msg.setWindowTitle("Thông báo")
                msg.setText("Thêm thành công")
                msg.setStandardButtons(QMessageBox.Ok)
                result = msg.exec_()
                row = 0
                self.fi_IDSV_2.setText("")
                self.fi_HoTenSV_2.setText("")
                self.fi_LopSV_2.setText("")
                self.fi_KhoaSV_2.setText("")
                self.lstSVLHP = []
                for row1 in cursor.execute(
                        f"select * from CHITIETLHP,SINHVIEN WHERE CHITIETLHP.MASV=SINHVIEN.MASV AND CHITIETLHP.MALOPMH='{self.fi_IDLop_2.text()}'"):
                    LHP = {}
                    LHP["ID"] = row1[2]
                    LHP["name"] = row1[3]
                    self.lstSVLHP.append(LHP)
                self.tbl_DSSV_2.setColumnWidth(0, 110)
                self.tbl_DSSV_2.setColumnWidth(1, 200)
                self.tbl_DSSV_2.setRowCount(len(self.lstSVLHP))
                for mh in self.lstSVLHP:
                    self.tbl_DSSV_2.setItem(row, 0, QtWidgets.QTableWidgetItem(mh["ID"]))
                    self.tbl_DSSV_2.setItem(row, 1, QtWidgets.QTableWidgetItem(mh["name"]))
                    row += 1
            except:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Thông báo")
                msg.setText("Lỗi")
                msg.setStandardButtons(QMessageBox.Ok)
                result = msg.exec_()
    def TimSVThemLop(self):
        for row in cursor.execute(f"select * from SINHVIEN,KHOA,LOP WHERE SINHVIEN.MASV='{self.txt_TimSV_2.text()}' "
                                  f"AND SINHVIEN.MALOP=LOP.MALOP AND LOP.MAKHOA=KHOA.MAKHOA"):
            self.fi_IDSV_2.setText(row[0])
            self.fi_HoTenSV_2.setText(row[1])
            self.fi_LopSV_2.setText(row[10])
            self.fi_KhoaSV_2.setText(row[15])
    def XoaLHP(self):
        if self.fi_IDLop_2.text()=="":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Thông báo")
            msg.setText("Vui lòng chọn lớp muốn xóa??")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        else:
            dssv = []
            for row in cursor.execute(f"select * from CHITIETLHP Where MALOPMH='{self.fi_IDLop_2.text()}'"):
                dssv.append(row[1])
            try:
                for i in dssv:
                    cursor.execute("DELETE CHITIETLHP WHERE MASV=?",i)
                    con.commit()
                cursor.execute("DELETE LOPMONHOC WHERE MALOPMH=?",self.fi_IDLop_2.text())
                con.commit()
                msg = QMessageBox()
                msg.setIcon(QMessageBox.NoIcon)
                msg.setWindowTitle("Thông báo")
                msg.setText("Xóa thành công")
                msg.setStandardButtons(QMessageBox.Ok)
                result = msg.exec_()
                self.loadLHP1(self.fi_IDMonHoc_2.text())
                self.loadMonHoc()
                self.KhoiTaoMH()
            except:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.NoIcon)
                msg.setWindowTitle("Thông báo")
                msg.setText("Lỗi!!!")
                msg.setStandardButtons(QMessageBox.Ok)
                result = msg.exec_()



    def loadLHP1(self,str):
        # load các lớp học phần
        row = 0
        self.lstLHP = []
        for row1 in cursor.execute(f"select * from LOPMONHOC,GIANGVIEN WHERE LOPMONHOC.MAGV=GIANGVIEN.MAGV AND LOPMONHOC.MAMH='{str}'"):
            LHP = {}
            LHP["ID"] = row1[0]
            LHP["name"] = row1[7]
            self.lstLHP.append(LHP)
        self.tbl_LopHocPhan_2.setColumnWidth(0, 90)
        self.tbl_LopHocPhan_2.setColumnWidth(1, 169)
        self.tbl_LopHocPhan_2.setRowCount(len(self.lstLHP))
        for mh in self.lstLHP:
            self.tbl_LopHocPhan_2.setItem(row, 0, QtWidgets.QTableWidgetItem(mh["ID"]))
            self.tbl_LopHocPhan_2.setItem(row, 1, QtWidgets.QTableWidgetItem(mh["name"]))
            row += 1
    def LuuLHP(self):
        if self.themlhp==True:
            if self.fi_IDLop_2.text()=="" or self.fi_SiSo_2.text()==""  or self.fi_GiangVien_2.currentIndex==-1:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Thông báo")
                msg.setText("Vui lòng nhập đầy đủ thông tin!!")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            else:
                if self.fi_IDMonHoc_2.text()=="":
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Vui lòng chọn môn học muốn thêm lớp")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                else:
                    try:
                        # bd=self.chuyenGio(self.fi_GioBD_22.text())
                        # kt = self.chuyenGio(self.fi_GioKT_22.text())
                        cursor.execute("SET DATEFORMAT DMY INSERT INTO LOPMONHOC "
                                       "Values(?,?,?,?,?,?)", self.fi_IDLop_2.text(),self.fi_IDMonHoc_2.text(),self.fi_NgayBatDau_2.text(),
                                       self.fi_NgayKetThuc_2.text(),self.fi_GiangVien_2.currentText().split("-")[1],self.fi_SiSo_2.text())
                        con.commit()
                        msg = QMessageBox()
                        msg.setIcon(QMessageBox.NoIcon)
                        msg.setWindowTitle("Thông báo")
                        msg.setText("Thêm lớp thành công")
                        msg.setStandardButtons(QMessageBox.Ok)
                        result = msg.exec_()
                        self.loadLHP1(self.fi_IDMonHoc_2.text())
                        self.loadMonHoc()
                        self.KhoiTaoMH()

                    except:
                        msg = QMessageBox()
                        msg.setIcon(QMessageBox.Warning)
                        msg.setWindowTitle("Thông báo")
                        msg.setText("Lỗi")
                        msg.setStandardButtons(QMessageBox.Ok)
                        result = msg.exec_()
        elif self.themlhp==False:
            if self.fi_IDLop_2.text()=="" or self.fi_SiSo_2.text()==""  or self.fi_GiangVien_2.currentIndex==-1:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Thông báo")
                msg.setText("Vui lòng nhập đầy đủ thông tin!!")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            else:
                if self.fi_IDMonHoc_2.text()=="":
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Vui lòng chọn môn học muốn lớp muốn cập nhật")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                else:
                    try:
                        # bd=self.chuyenGio(self.fi_GioBD_22.text())
                        # kt = self.chuyenGio(self.fi_GioKT_22.text())
                        cursor.execute("SET DATEFORMAT DMY UPDATE LOPMONHOC "
                                       "SET MAMH=?,NGAYBD=?,NGAYKT=?,MAGV=?, SISO=? WHERE MALOPMH=?",self.fi_IDMonHoc_2.text(),self.fi_NgayBatDau_2.text(),
                                       self.fi_NgayKetThuc_2.text(),
                                       self.fi_GiangVien_2.currentText().split("-")[1],self.fi_SiSo_2.text(),
                                       self.fi_IDLop_2.text())
                        con.commit()
                        msg = QMessageBox()
                        msg.setIcon(QMessageBox.NoIcon)
                        msg.setWindowTitle("Thông báo")
                        msg.setText("Cập nhật lớp thành công")
                        msg.setStandardButtons(QMessageBox.Ok)
                        result = msg.exec_()
                        self.loadLHP1(self.fi_IDMonHoc_2.text())
                        self.loadMonHoc()
                        self.KhoiTaoMH()

                    except:
                        msg = QMessageBox()
                        msg.setIcon(QMessageBox.Warning)
                        msg.setWindowTitle("Thông báo")
                        msg.setText("Lỗi")
                        msg.setStandardButtons(QMessageBox.Ok)
                        result = msg.exec_()
    def SuaLHP(self):
        if self.fi_IDLop_2.text()=="":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.NoIcon)
            msg.setWindowTitle("Thông báo")
            msg.setText("Vui lòng chọn lớp học cần cập nhật thông tin!!")
            msg.setStandardButtons(QMessageBox.Ok)
            result = msg.exec_()
            self.loadPageGV()
        else:
            self.themlhp = False
            self.btn_DatLichHoc_2.setVisible(False)
            self.fi_GiangVien_2.setEnabled(True)
            self.fi_SiSo_2.setEnabled(True)
            self.fi_NgayBatDau_2.setEnabled(True)
            self.fi_NgayKetThuc_2.setEnabled(True)
            # self.fi_GioBD_22.setEnabled(True)
            # self.fi_GioKT_22.setEnabled(True)
            # self.fi_PhongHoc_2.setEnabled(True)
            self.btn_ThemLopHP_2.setVisible(False)
            self.btn_SuaLopHP_2.setVisible(False)
            self.btn_XoaLopHP_2.setVisible(False)
            self.btn_LuuLHP_2.setVisible(True)
            self.btn_HuyLHP_2.setVisible(True)
    def HuyLHP(self):
        self.KhoiTaoMH()
    def ThemLHP(self):
        self.themlhp=True
        self.btn_DatLichHoc_2.setVisible(False)
        self.fi_GiangVien_2.setEnabled(True)
        self.fi_SiSo_2.setEnabled(True)
        self.fi_NgayBatDau_2.setEnabled(True)
        self.fi_NgayKetThuc_2.setEnabled(True)
        # self.fi_GioBD_22.setEnabled(True)
        # self.fi_GioKT_22.setEnabled(True)
        # self.fi_PhongHoc_2.setEnabled(True)
        self.btn_ThemLopHP_2.setVisible(False)
        self.btn_SuaLopHP_2.setVisible(False)
        self.btn_XoaLopHP_2.setVisible(False)
        self.btn_LuuLHP_2.setVisible(True)
        self.btn_HuyLHP_2.setVisible(True)
        self.fi_GiangVien_2.setCurrentIndex(-1)
        self.fi_GiangVien_2.setCurrentIndex(-1)
        self.fi_SiSo_2.setText("")
        # self.fi_PhongHoc_2.setCurrentIndex(-1)
        self.fi_NgayBatDau_2.setDate(QtCore.QDate(2000, 1, 1))
        self.fi_NgayKetThuc_2.setDate(QtCore.QDate(2000, 1, 1))
        # self.fi_GioBD_22.setTime(QtCore.QTime(12,0))
        # self.fi_GioKT_22.setTime(QtCore.QTime(12,0))

        sql = " EXEC LOPHP_ID "
        cursor.execute(sql)
        data = cursor.fetchall()
        self.fi_IDLop_2.setText(data[0][0])
    def kTXoaMH(self):
        cursor.execute(f'Select *from LOPMONHOC WHERE MAMH=?',self.fi_IDMonHoc_2.text())
        data = cursor.fetchall()
        if data==[]:
            return False
        else:
            return True
    def XoaMH(self):
        if self.fi_IDMonHoc_2.text()=="":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Thông báo")
            msg.setText("Vui lòng chọn môn học muốn xóa??")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        else:
            if self.kTXoaMH()==True:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Thông báo")
                msg.setText("Không thể xóa!!")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            elif self.kTXoaMH()==False:
                try:
                    print("n")
                    cursor.execute("DELETE MONHOC WHERE MAMH=?",self.fi_IDMonHoc_2.text())
                    con.commit()
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.NoIcon)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Xóa thành công")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()
                    self.loadMonHoc()
                    self.KhoiTaoMH()
                    self.fi_SoTC_2.setText("")
                    self.fi_TietLT_2.setText("")
                    self.fi_TenMonHoc_2.setText("")
                    self.fi_TongSoTiet_2.setText("")
                    self.fi_TietTH_2.setText("")
                    self.fi_IDMonHoc_2.setText("")
                except:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.NoIcon)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Lỗi!!!")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()



    def LuuMH(self):
        if self.themmh==True:
            if self.fi_SoTC_2.text()=="" or self.fi_TietLT_2.text()=="" or self.fi_TenMonHoc_2.text()=="" or self.fi_TongSoTiet_2.text()==""or self.fi_TietTH_2.text()=="":
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Thông báo")
                msg.setText("Vui lòng nhập đầy đủ thông tin")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            else:
                try:
                    cursor.execute("INSERT INTO MONHOC "
                                   "Values(?,?,?,?,?,?)", self.fi_IDMonHoc_2.text(), self.fi_TenMonHoc_2.text(),
                                   self.fi_SoTC_2.text(), self.fi_TongSoTiet_2.text(), self.fi_TietLT_2.text(),self.fi_TietTH_2.text())
                    con.commit()
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.NoIcon)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Thêm môn học thành công")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()
                    self.loadMonHoc()
                    self.KhoiTaoMH()
                except:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Lỗi")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()
        elif self.themmh==False:
            if self.fi_SoTC_2.text()=="" or self.fi_TietLT_2.text()=="" or self.fi_TenMonHoc_2.text()=="" or self.fi_TongSoTiet_2.text()==""or self.fi_TietTH_2.text()=="":
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Thông báo")
                msg.setText("Vui lòng nhập đầy đủ thông tin")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            else:
                try:
                    cursor.execute("UPDATE MONHOC "
                                   "SET TENMH=?, SOTC=?, TONGSOTIET=?, SOTIETLITHUYET=?,SOTIETTHUCHANH=? WHERE MONHOC.MAMH=?",  self.fi_TenMonHoc_2.text(),
                                   self.fi_SoTC_2.text(), self.fi_TongSoTiet_2.text(), self.fi_TietLT_2.text(),self.fi_TietTH_2.text(),self.fi_IDMonHoc_2.text())
                    con.commit()
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.NoIcon)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Cập nhật học thành công")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()
                    self.loadMonHoc()
                    self.KhoiTaoMH()
                except:
                    con.commit()
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Lỗi")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()
    def SuaMH(self):
        if self.fi_IDMonHoc_2.text()=="":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.NoIcon)
            msg.setWindowTitle("Thông báo")
            msg.setText("Vui lòng chọn  môn học cần cập nhật thông tin!!")
            msg.setStandardButtons(QMessageBox.Ok)
            result = msg.exec_()
            self.loadPageGV()
        else:
            self.themmh = False
            self.fi_SoTC_2.setEnabled(True)
            self.fi_TietLT_2.setEnabled(True)
            self.fi_TenMonHoc_2.setEnabled(True)
            self.fi_TongSoTiet_2.setEnabled(True)
            self.fi_TietTH_2.setEnabled(True)
            self.btn_LuuMH_2.setVisible(True)
            self.btn_HuyLopHoc_2.setVisible(True)
            self.btn_ThemMH_2.setVisible(False)
            self.btn_SuaMH_2.setVisible(False)
            self.btn_XoaLopHoc_2.setVisible(False)
    def HuyMH(self):
        self.KhoiTaoMH()
        self.fi_SoTC_2.setText("")
        self.fi_TietLT_2.setText("")
        self.fi_TenMonHoc_2.setText("")
        self.fi_TongSoTiet_2.setText("")
        self.fi_TietTH_2.setText("")
        self.fi_IDMonHoc_2.setText("")
    def ThemMH(self):
        self.themmh=True
        self.fi_SoTC_2.setEnabled(True)
        self.fi_TietLT_2.setEnabled(True)
        self.fi_TenMonHoc_2.setEnabled(True)
        self.fi_TongSoTiet_2.setEnabled(True)
        self.fi_TietTH_2.setEnabled(True)
        self.btn_LuuMH_2.setVisible(True)
        self.btn_HuyLopHoc_2.setVisible(True)
        self.btn_ThemMH_2.setVisible(False)
        self.btn_SuaMH_2.setVisible(False)
        self.btn_XoaLopHoc_2.setVisible(False)

        self.fi_SoTC_2.setText("")
        self.fi_TietLT_2.setText("")
        self.fi_TenMonHoc_2.setText("")
        self.fi_TongSoTiet_2.setText("")
        self.fi_TietTH_2.setText("")
        sql = " EXEC MONHOC_ID "
        cursor.execute(sql)
        data = cursor.fetchall()
        self.fi_IDMonHoc_2.setText(data[0][0])
    def KhoiTaoMH(self):
        self.btn_DatLichHoc_2.setVisible(True)
        self.fi_IDMonHoc_2.setEnabled(False)
        self.fi_SoTC_2.setEnabled(False)
        self.fi_TietLT_2.setEnabled(False)
        self.fi_TenMonHoc_2.setEnabled(False)
        self.fi_TongSoTiet_2.setEnabled(False)
        self.fi_TietTH_2.setEnabled(False)
        self.btn_LuuMH_2.setVisible(False)
        self.btn_HuyLopHoc_2.setVisible(False)
        self.fi_IDLop_2.setEnabled(False)
        self.fi_GiangVien_2.setEnabled(False)
        self.fi_GiangVien_2.setCurrentIndex(-1)
        self.fi_SiSo_2.setEnabled(False)
        self.fi_NgayBatDau_2.setEnabled(False)
        self.fi_NgayKetThuc_2.setEnabled(False)
        # self.fi_GioBD_22.setEnabled(False)
        # self.fi_GioKT_22.setEnabled(False)
        # self.fi_PhongHoc_2.setEnabled(False)
        self.fi_IDSV_2.setEnabled(False)
        self.fi_HoTenSV_2.setEnabled(False)
        self.fi_LopSV_2.setEnabled(False)
        self.fi_KhoaSV_2.setEnabled(False)
        self.btn_ThemMH_2.setVisible(True)
        self.btn_SuaMH_2.setVisible(True)
        self.btn_XoaLopHoc_2.setVisible(True)
        self.btn_XoaLopHP_2.setVisible(True)
        self.btn_SuaLopHP_2.setVisible(True)
        self.btn_ThemLopHP_2.setVisible(True)
        self.btn_LuuLHP_2.setVisible(False)
        self.btn_HuyLHP_2.setVisible(False)
        self.fi_IDLop_2.setText("")
        self.fi_GiangVien_2.setCurrentIndex(-1)
        self.fi_GiangVien_2.setCurrentIndex(-1)
        self.fi_SiSo_2.setText("")
        # self.fi_PhongHoc_2.setCurrentIndex(-1)
        self.fi_NgayBatDau_2.setDate(QtCore.QDate(2000, 1, 1))
        self.fi_NgayKetThuc_2.setDate(QtCore.QDate(2000, 1, 1))
        # self.fi_GioBD_22.setTime(QtCore.QTime(12, 0))
        # self.fi_GioKT_22.setTime(QtCore.QTime(12, 0))


    def GetSV_MonHoc(self):
        row = self.tbl_DSSV_2.currentIndex().row()
        ms = self.tbl_DSSV_2.item(row, 0).text()
        for row in cursor.execute(f"select * from SINHVIEN,KHOA,LOP,CHUYENNGANH WHERE SINHVIEN.MASV='{ms}' "
                                  f"AND LOP.MAKHOA=KHOA.MAKHOA AND SINHVIEN.MALOP=LOP.MALOP AND SINHVIEN.MACN=CHUYENNGANH.MACN "):
            self.fi_IDSV_2.setText(row[0])
            self.fi_HoTenSV_2.setText(row[1])
            self.fi_LopSV_2.setText(row[10])
            self.fi_KhoaSV_2.setText(row[15])
    def GetLopHocPhan(self):
        row = self.tbl_LopHocPhan_2.currentIndex().row()
        ms = self.tbl_LopHocPhan_2.item(row, 0).text()
        for row in cursor.execute(f"select * from LOPMONHOC,GIANGVIEN WHERE LOPMONHOC.MAGV=GIANGVIEN.MAGV AND MALOPMH='{ms}'"):
            self.fi_IDLop_2.setText(row[0])
            self.fi_GiangVien_2.setCurrentText(row[7]+"-"+row[6])
            self.fi_SiSo_2.setText(str(row[5]))
            self.fi_NgayBatDau_2.setDate(row[2])
            self.fi_NgayKetThuc_2.setDate(row[3])
            # self.fi_PhongHoc_2.setCurrentText(row[8])
            # self.fi_GioBD_22.setTime(QtCore.QTime(int(row[4].split(":")[0]),int(row[4].split(":")[1])))
            # self.fi_GioKT_22.setTime(QtCore.QTime(int(row[5].split(":")[0]), int(row[5].split(":")[1])))

        #load sv trong lớp học phần
        row = 0
        self.lstSVLHP = []
        for row1 in cursor.execute(f"select * from CHITIETLHP,SINHVIEN WHERE CHITIETLHP.MASV=SINHVIEN.MASV AND CHITIETLHP.MALOPMH='{ms}'"):
            LHP = {}
            LHP["ID"] = row1[2]
            LHP["name"] = row1[3]
            self.lstSVLHP.append(LHP)
        self.tbl_DSSV_2.setColumnWidth(0, 110)
        self.tbl_DSSV_2.setColumnWidth(1, 200)
        self.tbl_DSSV_2.setRowCount(len(self.lstSVLHP))
        for mh in self.lstSVLHP:
            self.tbl_DSSV_2.setItem(row, 0, QtWidgets.QTableWidgetItem(mh["ID"]))
            self.tbl_DSSV_2.setItem(row, 1, QtWidgets.QTableWidgetItem(mh["name"]))
            row += 1
    def GetMonHoc(self):
        row = self.tbl_MonHoc_2.currentIndex().row()
        ms = self.tbl_MonHoc_2.item(row, 0).text()
        for row in cursor.execute(f"select * from MONHOC Where MAMH='{ms}'"):
            self.fi_IDMonHoc_2.setText(row[0])
            self.fi_TenMonHoc_2.setText(row[1])
            self.fi_SoTC_2.setText(str(row[2]))
            self.fi_TongSoTiet_2.setText(str(row[3]))
            self.fi_TietLT_2.setText(str(row[4]))
            self.fi_TietTH_2.setText(str(row[5]))


        #load các lớp học phần
        row = 0
        self.lstLHP = []
        for row1 in cursor.execute(f"select * from LOPMONHOC,GIANGVIEN WHERE LOPMONHOC.MAGV=GIANGVIEN.MAGV AND LOPMONHOC.MAMH='{ms}'"):
            LHP = {}
            LHP["ID"] = row1[0]
            LHP["name"] = row1[7]
            self.lstLHP.append(LHP)
        self.tbl_LopHocPhan_2.setColumnWidth(0, 90)
        self.tbl_LopHocPhan_2.setColumnWidth(1, 169)
        self.tbl_LopHocPhan_2.setRowCount(len(self.lstLHP))
        for mh in self.lstLHP:
            self.tbl_LopHocPhan_2.setItem(row, 0, QtWidgets.QTableWidgetItem(mh["ID"]))
            self.tbl_LopHocPhan_2.setItem(row, 1, QtWidgets.QTableWidgetItem(mh["name"]))
            row += 1


    def loadMonHoc(self):
        #load combobox giảng viên
        self.fi_GiangVien_2.clear()
        for row1 in cursor.execute("select * from GIANGVIEN"):
            self.fi_GiangVien_2.addItem(row1[1]+'-'+row1[0])
        # self.fi_PhongHoc_2.clear()
        # for row1 in cursor.execute("select * from PHONGHOC"):
        #     self.fi_PhongHoc_2.addItem(row1[0])
        #load môn học
        row = 0
        self.lstMH = []
        for row1 in cursor.execute("select * from MONHOC"):
            MH = {}
            MH["ID"] = row1[0]
            MH["name"] = row1[1]
            MH["soTC"] = row1[2]
            self.lstMH.append(MH)
        self.tbl_MonHoc_2.setColumnWidth(0, 150)
        self.tbl_MonHoc_2.setColumnWidth(1, 280)
        self.tbl_MonHoc_2.setColumnWidth(2, 225)
        self.tbl_MonHoc_2.setRowCount(len(self.lstMH))
        for mh in self.lstMH:
            self.tbl_MonHoc_2.setItem(row, 0, QtWidgets.QTableWidgetItem(mh["ID"]))
            self.tbl_MonHoc_2.setItem(row, 1, QtWidgets.QTableWidgetItem(mh["name"]))
            self.tbl_MonHoc_2.setItem(row, 2, QtWidgets.QTableWidgetItem(str(mh["soTC"])))
            row += 1
        self.cbo_TimMH_2.clear()
        self.cbo_TimMH_2.addItem("ID")
        self.cbo_TimMH_2.addItem("Tên Môn Học")
    #page sinh viên ##############################################################################################
    def Training(self):
        folder = ('Data/processed/')
        database = {}
        cnt=0
        self.progressBar.setVisible(True)
        self.btn_TrainingData.setEnabled(False)
        for filename in os.listdir(folder):
            path = folder + filename
            print(path)
            if cnt < 88:
                self.progressBar.setValue(cnt)
            else:
                self.progressBar.setValue(88)
            cnt+=3
            QApplication.processEvents()
            for filename2 in os.listdir(path):
                gbr1 = cv2.imread(path + "/" + filename2)
                face = cv2.resize(gbr1, (160, 160))
                face = face.astype('float32')
                mean, std = face.mean(), face.std()
                face = (face - mean) / std
                face = expand_dims(face, axis=0)
                signature = MyFaceNet.predict(face)
                database[os.path.splitext(filename)[0]] = signature

        myfile = open("data.pkl", "wb")
        pickle.dump(database, myfile)
        myfile.close()
        self.progressBar.setValue(100)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.NoIcon)
        msg.setWindowTitle("Thông báo")
        msg.setText("Training data thành công")
        msg.setStandardButtons(QMessageBox.Ok)
        result = msg.exec_()
        self.progressBar.setVisible(False)
        self.btn_TrainingData.setEnabled(True)
    def LayAnh(self):
        if self.kTLuuSV() == False:
            if self.fi_IDSV.text() == "":
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Thông báo")
                msg.setText("Vui lòng chọn sinh viên muốn thêm ảnh")
                msg.setStandardButtons(QMessageBox.Ok )
                result = msg.exec_()
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Thông báo")
                msg.setText("Bạn có muốn lưu sinh viên " + self.fi_IDSV.text() + " không?")
                msg.setStandardButtons(QMessageBox.Ok | QMessageBox.No)
                result = msg.exec_()
                if result == QMessageBox.Ok:
                    self.them = True
                    self.LuuSV()
                    # msg = QMessageBox()
                    # msg.setIcon(QMessageBox.Warning)
                    # msg.setWindowTitle("Thông báo")
                    # msg.setText("Lưu thành công")
                    # msg.setStandardButtons(QMessageBox.Ok | QMessageBox.No)
                    # result = msg.exec_()
                    self.anhSV.fi_ID.setText(self.fi_IDSV.text())
                    self.anhSV.fi_HoTen.setText(self.fi_HoTen.text())
                    self.anhSV.fi_ChuyenNganh.setText(self.fi_ChuyenNganh.currentText())
                    self.anhSV.fi_Lop.setText(self.fi_Lop.currentText())
                    self.anhSV.show()
                    self.hide()
        else:
            self.anhSV.fi_ID.setText(self.fi_IDSV.text())
            self.anhSV.fi_HoTen.setText(self.fi_HoTen.text())
            self.anhSV.fi_ChuyenNganh.setText(self.fi_ChuyenNganh.currentText())
            self.anhSV.fi_Lop.setText(self.fi_Lop.currentText())
            self.anhSV.show()
            self.hide()
    def kTLuuSV(self):
        try:
            cursor.execute(f"select* from SINHVIEN Where MASV='{self.fi_IDSV.text()}'")
            data = cursor.fetchall()
            if data[0][0] != None:
                return True
        except:
            pass
        return False

    def XoaLop(self):
        if self.fi_TenLopHoc.text() == "":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Thông báo")
            msg.setText("Vui lòng chọn Lớp muốn xóa")
            msg.setStandardButtons(QMessageBox.Ok)
            result = msg.exec_()
        else:
            if self.kTLopHocTonTaiSV() == True:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.NoIcon)
                msg.setWindowTitle("Thông báo")
                msg.setText("Không thể xóa!!!")
                msg.setStandardButtons(QMessageBox.Ok)
                result = msg.exec_()
            else:
                try:
                    cursor.execute("DELETE LOP WHERE MALOP=?", self.fi_TenLopHoc.text())
                    con.commit()
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.NoIcon)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Xóa thành công")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()
                    self.fi_TenLopHoc.setText("")
                    self.loadData()
                    self.KhoiTao()
                    self.KhoiTaoCombobox()
                except:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.NoIcon)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Lỗi!!!")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()

    def kTLopHocTonTaiSV(self):
        try:
            cursor.execute(f"select* from SINHVIEN Where MALOP='{self.fi_TenLopHoc.text()}'")
            data = cursor.fetchall()
            if data[0][0] != None:
                return True
        except:
            pass
        return False

    def kTLopHoc(self, tenlop):
        try:
            cursor.execute(f"select* from LOP Where MALOP='{self.fi_TenLopHoc.text()}'")
            data = cursor.fetchall()
            if data[0][0] != None:
                return True
        except:
            pass
        return False

    def LuuLop(self):
        if self.themlop == True:
            if self.fi_TenLopHoc.text() == "" or self.fi_KhoaLop.currentIndex == -1:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Thông báo")
                msg.setText("Vui lòng nhập đầy đủ thông tin")
                msg.setStandardButtons(QMessageBox.Ok)
                result = msg.exec_()
            else:
                if self.kTLopHoc(self.fi_TenLopHoc.text()) == True:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Lớp học " + self.fi_TenLopHoc.text() + " đã tồn tại. Vui lòng nhập lớp học mới!!")
                    msg.setStandardButtons(QMessageBox.Ok )
                    result = msg.exec_()
                else:
                    try:
                        cursor.execute("insert into LOP VALUES(?,?)", self.fi_TenLopHoc.text(),
                                       self.fi_KhoaLop.currentText().split('-')[1])
                        con.commit()
                        msg = QMessageBox()
                        msg.setIcon(QMessageBox.NoIcon)
                        msg.setWindowTitle("Thông báo")
                        msg.setText("Thêm thành công")
                        msg.setStandardButtons(QMessageBox.Ok)
                        result = msg.exec_()
                        self.loadData()
                        self.btn_ThemLop.setVisible(True)
                        self.btn_XoaLop.setVisible(True)
                        self.btn_SuaLop.setVisible(True)
                        self.btn_LuuLop.setVisible(False)
                        self.btn_HuyLop.setVisible(False)
                        self.fi_KhoaLop.setEnabled(False)
                        self.fi_TenLopHoc.setEnabled(False)
                    except:
                        msg = QMessageBox()
                        msg.setIcon(QMessageBox.Warning)
                        msg.setWindowTitle("Thông báo")
                        msg.setText("Lỗi")
                        msg.setStandardButtons(QMessageBox.Ok)
                        result = msg.exec_()


        elif self.themlop == False:
            if self.fi_TenLopHoc.text() == "" or self.fi_KhoaLop.currentIndex == -1:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Thông báo")
                msg.setText("Vui lòng nhập đầy đủ thông tin")
                msg.setStandardButtons(QMessageBox.Ok)
                result = msg.exec_()
            else:
                try:
                    cursor.execute("UPDATE LOP SET MAKHOA=? WHERE MALOP=?", self.fi_KhoaLop.currentText().split('-')[1],
                                   self.fi_TenLopHoc.text())
                    con.commit()
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.NoIcon)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Cập nhật thành công")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()
                    self.loadData()
                    self.btn_ThemLop.setVisible(True)
                    self.btn_XoaLop.setVisible(True)
                    self.btn_SuaLop.setVisible(True)
                    self.btn_LuuLop.setVisible(False)
                    self.btn_HuyLop.setVisible(False)
                    self.fi_KhoaLop.setEnabled(False)
                    self.fi_TenLopHoc.setEnabled(False)
                except:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Lỗi")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()

    def HuyLop(self):
        self.btn_ThemLop.setVisible(True)
        self.btn_XoaLop.setVisible(True)
        self.btn_SuaLop.setVisible(True)
        self.btn_LuuLop.setVisible(False)
        self.btn_HuyLop.setVisible(False)
        self.fi_KhoaLop.setEnabled(False)
        self.fi_KhoaLop.setCurrentIndex(-1)
        self.fi_TenLopHoc.setEnabled(False)
        self.fi_TenLopHoc.setText("")

    def SuaLop(self):
        if self.fi_TenLopHoc.text()=="":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.NoIcon)
            msg.setWindowTitle("Thông báo")
            msg.setText("Vui lòng chọn lớp học cần cập nhật thông tin!!")
            msg.setStandardButtons(QMessageBox.Ok)
            result = msg.exec_()
            self.loadPageGV()
        else:
            self.themlop = False
            self.fi_KhoaLop.setEnabled(True)
            self.btn_ThemLop.setVisible(False)
            self.btn_XoaLop.setVisible(False)
            self.btn_SuaLop.setVisible(False)
            self.btn_LuuLop.setVisible(True)
            self.btn_HuyLop.setVisible(True)

    def ThemLop(self):
        self.themlop = True
        self.fi_TenLopHoc.setEnabled(True)
        self.fi_KhoaLop.setEnabled(True)
        self.btn_ThemLop.setVisible(False)
        self.btn_XoaLop.setVisible(False)
        self.btn_SuaLop.setVisible(False)
        self.btn_LuuLop.setVisible(True)
        self.btn_HuyLop.setVisible(True)
        self.fi_TenLopHoc.setText("")
        self.fi_KhoaLop.setCurrentIndex(-1)

    def XoaSV(self):
        if self.fi_IDSV.text() == "":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Thông báo")
            msg.setText("Vui lòng chọn Sinh viên muốn xóa")
            msg.setStandardButtons(QMessageBox.Ok)
            result = msg.exec_()
        else:
            try:
                cursor.execute("DELETE SINHVIEN WHERE MASV=?", self.fi_IDSV.text())
                con.commit()
                msg = QMessageBox()
                msg.setIcon(QMessageBox.NoIcon)
                msg.setWindowTitle("Thông báo")
                msg.setText("Xóa sinh viên thành công")
                msg.setStandardButtons(QMessageBox.Ok )
                result = msg.exec_()
                self.fi_IDSV.setText("")
                self.loadData()
                self.KhoiTao()
                self.LamMoiSV()
                self.KhoiTaoCombobox()
            except:
                con.commit()
                msg = QMessageBox()
                msg.setIcon(QMessageBox.NoIcon)
                msg.setWindowTitle("Thông báo")
                msg.setText("Lỗi!!!")
                msg.setStandardButtons(QMessageBox.Ok)
                result = msg.exec_()

    def KhoiTaoCombobox(self):
        self.fi_Khoa.setCurrentIndex(-1)
        self.fi_ChuyenNganh.setCurrentIndex(-1)
        self.fi_NienKhoa.setCurrentIndex(-1)
        self.fi_Lop.setCurrentIndex(-1)
        self.fi_GioiTinh.setCurrentIndex(-1)
        self.fi_KhoaLop.setCurrentIndex(-1)

    def SuaSV(self):
        if self.fi_IDSV.text()=="":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.NoIcon)
            msg.setWindowTitle("Thông báo")
            msg.setText("Vui lòng chọn sinh viên cần cập nhật thông tin!!")
            msg.setStandardButtons(QMessageBox.Ok)
            result = msg.exec_()
            self.loadPageGV()
        else:
            self.them = False
            self.fi_Khoa.setEnabled(True)
            self.fi_ChuyenNganh.setEnabled(True)
            self.fi_NienKhoa.setEnabled(True)
            self.fi_Lop.setEnabled(True)
            self.fi_GioiTinh.setEnabled(True)
            self.fi_DCThuongTru.setEnabled(True)
            self.fi_DCTamTru.setEnabled(True)
            self.fi_HoTen.setEnabled(True)
            self.fi_CMND.setEnabled(True)
            self.fi_NgaySinh.setEnabled(True)
            self.fi_SDT.setEnabled(True)
            self.fi_DanToc.setEnabled(True)
            self.fi_Email_2.setEnabled(True)
            self.fi_TenLopHoc.setEnabled(True)
            self.fi_KhoaLop.setEnabled(True)
            self.btnLuuSV.setVisible(True)
            self.btnHuySV.setVisible(True)
            self.btn_LamMoiSV.setVisible(True)
            self.btn_ThemSV.setVisible(False)
            self.btn_SuaSV.setVisible(False)
            self.btn_XoaSV.setVisible(False)

    def kiemTraNhapSV(self):
        if self.fi_Khoa.currentIndex() == -1 or self.fi_ChuyenNganh.currentIndex() == -1 or self.fi_NienKhoa.currentIndex() == -1 or self.fi_Lop.currentIndex() == -1 or self.fi_GioiTinh.currentIndex() == -1 or self.fi_IDSV.text() == "" or self.fi_DCThuongTru.text() == "" or self.fi_DCTamTru.text() == "" or self.fi_HoTen.text() == "" or self.fi_CMND.text() == "" or self.fi_SDT.text() == "" or self.fi_DanToc.text() == "" or self.fi_Email_2.text() == "":
            return False
        return True

    def LuuSV(self):
        if self.them == True:
            if self.kiemTraNhapSV() == False:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Thông báo")
                msg.setText("Vui lòng nhập đầy đủ thông tin")
                msg.setStandardButtons(QMessageBox.Ok)
                result = msg.exec_()
            else:
                try:
                    cursor.execute("set dateformat dmy "
                                   "INSERT INTO SINHVIEN(MASV,TENSV,SDT, NGAYSINH ,CMND,DANTOC,GIOITINH, EMAIL,DIACHITT ,DCTAMTRU ,MALOP ,MACN, MANK ) "
                                   "Values(?,?,?,?,?,?,?,?,?,?,?,?,?)", self.fi_IDSV.text(), self.fi_HoTen.text(),
                                   self.fi_SDT.text(),
                                   self.fi_NgaySinh.text(), self.fi_CMND.text(), self.fi_DanToc.text(),
                                   self.fi_GioiTinh.currentText(),
                                   self.fi_Email_2.text(), self.fi_DCThuongTru.text(), self.fi_DCTamTru.text(),
                                   self.fi_Lop.currentText(), self.fi_ChuyenNganh.currentText().split('-')[1],
                                   self.fi_NienKhoa.currentText().split('-')[2])
                    con.commit()
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.NoIcon)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Thêm sinh viên thành công")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()
                    self.loadData()
                    self.KhoiTao()
                except:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Lỗi")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()

        elif self.them == False:
            if self.kiemTraNhapSV() == False:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Thông báo")
                msg.setText("Vui lòng nhập đầy đủ thông tin")
                msg.setStandardButtons(QMessageBox.Ok )
                result = msg.exec_()
            else:
                if self.fi_CMND.text() == "":
                    print("t")
                try:
                    cursor.execute("set dateformat dmy "
                                   "Update SINHVIEN set TENSV=?,SDT=?, NGAYSINH=? ,CMND=?,DANTOC=?,GIOITINH=?, EMAIL=?,DIACHITT=?,DCTAMTRU=? ,MALOP=? ,MACN=?, MANK=? where MASV=? "
                                   , self.fi_HoTen.text(), self.fi_SDT.text(),
                                   self.fi_NgaySinh.text(), self.fi_CMND.text(), self.fi_DanToc.text(),
                                   self.fi_GioiTinh.currentText(),
                                   self.fi_Email_2.text(), self.fi_DCThuongTru.text(), self.fi_DCTamTru.text(),
                                   self.fi_Lop.currentText(), self.fi_ChuyenNganh.currentText().split('-')[1],
                                   self.fi_NienKhoa.currentText().split('-')[2], self.fi_IDSV.text(), )
                    con.commit()
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.NoIcon)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Cập nhật sinh viên thành công")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()
                    self.loadData()
                    self.KhoiTao()
                except:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Thông báo")
                    msg.setText("Lỗi")
                    msg.setStandardButtons(QMessageBox.Ok)
                    result = msg.exec_()

    def HuySV(self):
        self.KhoiTao()
        self.LamMoiSV()
        self.fi_IDSV.setText("")

    def LamMoiSV(self):
        self.fi_Khoa.setCurrentIndex(-1)
        self.fi_ChuyenNganh.setCurrentIndex(-1)
        self.fi_NienKhoa.setCurrentIndex(-1)

        self.fi_Lop.setCurrentIndex(-1)
        self.fi_GioiTinh.setCurrentIndex(-1)
        self.fi_DCThuongTru.setText("")
        self.fi_DCTamTru.setText("")
        self.fi_HoTen.setText("")
        self.fi_CMND.setText("")
        self.fi_NgaySinh.setDate(QtCore.QDate(2000, 1, 1))
        self.fi_SDT.setText("")
        self.fi_DanToc.setText("")
        self.fi_Email_2.setText("")
        image = np.array(Image.open(f"Avatar/Who1.png"))
        image = cv2.resize(image, (212, 137))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0],
                           QtGui.QImage.Format_RGB888)  # tổng số pixel quét của ảnh
        self.lbl_Avatar.setPixmap(QtGui.QPixmap.fromImage(img))

    def KhoiTao(self):
        self.fi_Khoa.setEnabled(False)
        self.fi_ChuyenNganh.setEnabled(False)
        self.fi_NienKhoa.setEnabled(False)
        self.fi_IDSV.setEnabled(False)
        self.fi_Lop.setEnabled(False)
        self.fi_GioiTinh.setEnabled(False)
        self.fi_DCThuongTru.setEnabled(False)
        self.fi_DCTamTru.setEnabled(False)
        self.fi_HoTen.setEnabled(False)
        self.fi_CMND.setEnabled(False)
        self.fi_NgaySinh.setEnabled(False)
        self.fi_SDT.setEnabled(False)
        self.fi_DanToc.setEnabled(False)
        self.fi_Email_2.setEnabled(False)
        self.fi_TenLopHoc.setEnabled(False)
        self.fi_KhoaLop.setEnabled(False)
        self.btnLuuSV.setVisible(False)
        self.btnHuySV.setVisible(False)
        self.btn_LuuLop.setVisible(False)
        self.btn_HuyLop.setVisible(False)
        self.btn_LamMoiSV.setVisible(False)
        self.btn_ThemSV.setVisible(True)
        self.btn_SuaSV.setVisible(True)
        self.btn_XoaSV.setVisible(True)
        self.rbo_CoAnh.setEnabled(False)
        self.rbo_KhongAnh.setEnabled(False)
        image = np.array(Image.open(f"Avatar/Who1.png"))
        image = cv2.resize(image, (212, 137))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0],
                           QtGui.QImage.Format_RGB888)  # tổng số pixel quét của ảnh
        self.lbl_Avatar.setPixmap(QtGui.QPixmap.fromImage(img))

    def themSV(self):
        self.them = True
        self.fi_Khoa.setEnabled(True)
        self.fi_ChuyenNganh.setEnabled(True)
        self.fi_NienKhoa.setEnabled(True)
        self.fi_Lop.setEnabled(True)
        self.fi_GioiTinh.setEnabled(True)
        self.fi_DCThuongTru.setEnabled(True)
        self.fi_DCTamTru.setEnabled(True)
        self.fi_HoTen.setEnabled(True)
        self.fi_CMND.setEnabled(True)
        self.fi_NgaySinh.setEnabled(True)
        self.fi_SDT.setEnabled(True)
        self.fi_DanToc.setEnabled(True)
        self.fi_Email_2.setEnabled(True)
        self.fi_TenLopHoc.setEnabled(True)
        self.fi_KhoaLop.setEnabled(True)
        self.btnLuuSV.setVisible(True)
        self.btnHuySV.setVisible(True)
        self.btn_LamMoiSV.setVisible(True)
        self.btn_ThemSV.setVisible(False)
        self.btn_SuaSV.setVisible(False)
        self.btn_XoaSV.setVisible(False)
        self.LamMoiSV()
        sql = " EXEC SINHVIEN_ID "
        cursor.execute(sql)
        data = cursor.fetchall()
        self.fi_IDSV.setText(data[0][0])

    def GetAllLop(self):
        i = 0
        self.tbl_Lop.setColumnWidth(0, 95)
        self.tbl_Lop.setColumnWidth(1, 63)
        self.tbl_Lop.setRowCount(len(self.lstLop))
        for lop in self.lstLop:
            self.tbl_Lop.setItem(i, 0, QtWidgets.QTableWidgetItem(lop["ID"]))
            self.tbl_Lop.setItem(i, 1, QtWidgets.QTableWidgetItem(lop["Khoa"]))
            i += 1

    def TimLop(self):
        lstTimLop = []
        i = 0
        if self.txt_TimLop.text() == "":
            self.GetAllLop()
        elif self.cbo_TimLop.currentText() == "Lớp":
            for row in cursor.execute(f"select * from LOP,KHOA Where LOP.MAKHOA=KHOA.MAKHOA "
                                      f"and LOP.MALOP='{self.txt_TimLop.text()}' "):
                lop = {}
                lop["ID"] = row[0]
                lop["Khoa"] = row[2]
                lstTimLop.append(lop)
            self.tbl_Lop.setColumnWidth(0, 95)
            self.tbl_Lop.setColumnWidth(1, 63)
            self.tbl_Lop.setRowCount(len(lstTimLop))
            for lop in lstTimLop:
                self.tbl_Lop.setItem(i, 0, QtWidgets.QTableWidgetItem(lop["ID"]))
                self.tbl_Lop.setItem(i, 1, QtWidgets.QTableWidgetItem(lop["Khoa"]))
                i += 1

        elif self.cbo_TimLop.currentText() == "Khoa":
            for row in cursor.execute(f"select * from LOP,KHOA Where LOP.MAKHOA=KHOA.MAKHOA "
                                      f"and KHOA.MAKHOA='{self.txt_TimLop.text()}'"):
                lop = {}
                lop["ID"] = row[0]
                lop["Khoa"] = row[2]
                lstTimLop.append(lop)
            self.tbl_Lop.setColumnWidth(0, 95)
            self.tbl_Lop.setColumnWidth(1, 63)
            self.tbl_Lop.setRowCount(len(lstTimLop))
            for lop in lstTimLop:
                self.tbl_Lop.setItem(i, 0, QtWidgets.QTableWidgetItem(lop["ID"]))
                self.tbl_Lop.setItem(i, 1, QtWidgets.QTableWidgetItem(lop["Khoa"]))
                i += 1
        else:
            self.GetAllLop()

    def TimSV(self):
        lstTimSV = []
        row = 0
        if self.txt_TimSV.text() == "":
            self.GetAllSV()
        elif self.cbo_TimSV.currentText() == "ID Sinh viên":
            for row1 in cursor.execute(f"select * from SINHVIEN,LOP WHERE SINHVIEN.MALOP=LOP.MALOP"
                                       f" AND SINHVIEN.MASV='{self.txt_TimSV.text()}'"):
                SV = {}
                SV["ID"] = row1[0]
                SV["name"] = row1[1]
                SV["class"] = row1[14]
                lstTimSV.append(SV)

            self.tbl_SinhVien.setColumnWidth(0, 100)
            self.tbl_SinhVien.setColumnWidth(1, 240)
            self.tbl_SinhVien.setColumnWidth(2, 220)
            self.tbl_SinhVien.setRowCount(len(lstTimSV))
            for sv in lstTimSV:
                self.tbl_SinhVien.setItem(row, 0, QtWidgets.QTableWidgetItem(sv["ID"]))
                self.tbl_SinhVien.setItem(row, 1, QtWidgets.QTableWidgetItem(sv["name"]))
                self.tbl_SinhVien.setItem(row, 2, QtWidgets.QTableWidgetItem(sv["class"]))
                row += 1


        elif self.cbo_TimSV.currentText() == "Họ tên":
            for row1 in cursor.execute(f"select * from SINHVIEN,LOP WHERE SINHVIEN.MALOP=LOP.MALOP"
                                       f" AND SINHVIEN.TENSV LIKE N'%{self.txt_TimSV.text()}%'"):
                SV = {}
                SV["ID"] = row1[0]
                SV["name"] = row1[1]
                SV["class"] = row1[14]
                lstTimSV.append(SV)

            self.tbl_SinhVien.setColumnWidth(0, 100)
            self.tbl_SinhVien.setColumnWidth(1, 240)
            self.tbl_SinhVien.setColumnWidth(2, 220)
            self.tbl_SinhVien.setRowCount(len(lstTimSV))
            for sv in lstTimSV:
                self.tbl_SinhVien.setItem(row, 0, QtWidgets.QTableWidgetItem(sv["ID"]))
                self.tbl_SinhVien.setItem(row, 1, QtWidgets.QTableWidgetItem(sv["name"]))
                self.tbl_SinhVien.setItem(row, 2, QtWidgets.QTableWidgetItem(sv["class"]))
                row += 1


        elif self.cbo_TimSV.currentText() == "Lớp":
            for row1 in cursor.execute(f"select * from SINHVIEN,LOP WHERE SINHVIEN.MALOP=LOP.MALOP"
                                       f" AND LOP.MALOP='{self.txt_TimSV.text()}'"):
                SV = {}
                SV["ID"] = row1[0]
                SV["name"] = row1[1]
                SV["class"] = row1[14]
                lstTimSV.append(SV)

            self.tbl_SinhVien.setColumnWidth(0, 100)
            self.tbl_SinhVien.setColumnWidth(1, 240)
            self.tbl_SinhVien.setColumnWidth(2, 220)
            self.tbl_SinhVien.setRowCount(len(lstTimSV))
            for sv in lstTimSV:
                self.tbl_SinhVien.setItem(row, 0, QtWidgets.QTableWidgetItem(sv["ID"]))
                self.tbl_SinhVien.setItem(row, 1, QtWidgets.QTableWidgetItem(sv["name"]))
                self.tbl_SinhVien.setItem(row, 2, QtWidgets.QTableWidgetItem(sv["class"]))
                row += 1
        else:
            self.GetAllSV()

    def GetAllSV(self):
        row = 0
        self.tbl_SinhVien.setColumnWidth(0, 100)
        self.tbl_SinhVien.setColumnWidth(1, 240)
        self.tbl_SinhVien.setColumnWidth(2, 220)
        self.tbl_SinhVien.setRowCount(len(self.lstSV))
        for sv in self.lstSV:
            self.tbl_SinhVien.setItem(row, 0, QtWidgets.QTableWidgetItem(sv["ID"]))
            self.tbl_SinhVien.setItem(row, 1, QtWidgets.QTableWidgetItem(sv["name"]))
            self.tbl_SinhVien.setItem(row, 2, QtWidgets.QTableWidgetItem(sv["class"]))
            row += 1

    def GetLop(self):
        row = self.tbl_Lop.currentIndex().row()
        ms = self.tbl_Lop.item(row, 0).text()
        for row in cursor.execute(f"select*from LOP,KHOA WHERE LOP.MAKHOA=KHOA.MAKHOA AND LOP.MALOP='{ms}'"):
            self.fi_TenLopHoc.setText(row[0])
            self.fi_KhoaLop.setCurrentText(row[3] + "-" + row[2])

    def GetSV(self):
        row = self.tbl_SinhVien.currentIndex().row()
        ms = self.tbl_SinhVien.item(row, 0).text()
        try:
            for row in cursor.execute(f"select * from SINHVIEN,KHOA,LOP,CHUYENNGANH,NIENKHOA WHERE SINHVIEN.MASV='{ms}' "
                                      f"AND LOP.MAKHOA=KHOA.MAKHOA AND SINHVIEN.MALOP=LOP.MALOP AND SINHVIEN.MACN=CHUYENNGANH.MACN "
                                      f"AND NIENKHOA.MANK=SINHVIEN.MANK"):
                self.fi_IDSV.setText(row[0])
                self.fi_Khoa.setCurrentText(row[15] + "-" + row[14])
                self.fi_ChuyenNganh.setCurrentText(row[19] + "-" + row[18])
                self.fi_NienKhoa.setCurrentText(row[22] + '-' + row[21])
                self.fi_Lop.setCurrentText(row[16])
                self.fi_GioiTinh.setCurrentText(row[6])
                self.fi_Email_2.setText(row[7])
                self.fi_DCTamTru.setText(row[9])
                self.fi_DCThuongTru.setText(row[8])
                self.fi_DanToc.setText(row[5])
                self.fi_HoTen.setText(row[1])
                self.fi_CMND.setText(row[4])
                self.fi_SDT.setText(row[2])
                self.fi_NgaySinh.setDate(row[3])
                if row[11] == None:
                    self.rbo_KhongAnh.setChecked(True)
                    image = np.array(Image.open(f"Avatar/Who1.png"))
                    image = cv2.resize(image, (212, 137))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    img = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0],
                                       QtGui.QImage.Format_RGB888)  # tổng số pixel quét của ảnh
                    self.lbl_Avatar.setPixmap(QtGui.QPixmap.fromImage(img))
                else:
                    self.rbo_CoAnh.setChecked(True)
                    image = np.array(Image.open(f"Avatar/{row[11]}"))

                    image = cv2.resize(image, (212, 137))
                    img = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0],
                                       QtGui.QImage.Format_RGB888)  # tổng số pixel quét của ảnh
                    self.lbl_Avatar.setPixmap(QtGui.QPixmap.fromImage(img))
        except:
            pass

    def loadData(self):
        self.fi_ChuyenNganh.clear()
        self.fi_NienKhoa.clear()
        self.fi_Khoa.clear()
        self.fi_KhoaLop.clear()
        self.fi_Lop.clear()
        self.fi_GioiTinh.clear()
        self.cbo_TimSV.clear()
        self.cbo_TimLop.clear()
        for row in cursor.execute("select * from CHUYENNGANH"):
            self.fi_ChuyenNganh.addItem(row[1] + "-" + row[0])
        for row in cursor.execute("select * from NIENKHOA"):
            self.fi_NienKhoa.addItem(row[1] + "-" + row[0])
        for row in cursor.execute("select * from KHOA"):
            self.fi_Khoa.addItem(row[1] + "-" + row[0])
            self.fi_KhoaLop.addItem(row[1] + "-" + row[0])
        for row in cursor.execute("select * from LOP"):
            self.fi_Lop.addItem(row[0])

        self.fi_GioiTinh.addItem("Nam")
        self.fi_GioiTinh.addItem("Nữ")
        self.fi_GioiTinh.addItem("Khác")

        # combobox tìm kiếm sinh viên
        self.cbo_TimSV.addItem("ID Sinh viên")
        self.cbo_TimSV.addItem("Họ tên")
        self.cbo_TimSV.addItem("Lớp")

        # combobox tìm lớp
        self.cbo_TimLop.addItem("Lớp")
        self.cbo_TimLop.addItem("Khoa")

        # table Lop
        i = 0
        self.lstLop = []
        for row in cursor.execute("select * from LOP,KHOA Where LOP.MAKHOA=KHOA.MAKHOA "):
            lop = {}
            lop["ID"] = row[0]
            lop["Khoa"] = row[2]
            self.lstLop.append(lop)
        self.tbl_Lop.setColumnWidth(0, 95)
        self.tbl_Lop.setColumnWidth(1, 63)
        self.tbl_Lop.setRowCount(len(self.lstLop))
        for lop in self.lstLop:
            self.tbl_Lop.setItem(i, 0, QtWidgets.QTableWidgetItem(lop["ID"]))
            self.tbl_Lop.setItem(i, 1, QtWidgets.QTableWidgetItem(lop["Khoa"]))
            i += 1

        # table Sinh Viên
        row = 0
        self.lstSV = []
        for row1 in cursor.execute("select * from SINHVIEN,LOP WHERE SINHVIEN.MALOP=LOP.MALOP"):
            SV = {}
            SV["ID"] = row1[0]
            SV["name"] = row1[1]
            SV["class"] = row1[14]
            self.lstSV.append(SV)

        self.tbl_SinhVien.setColumnWidth(0, 100)
        self.tbl_SinhVien.setColumnWidth(1, 240)
        self.tbl_SinhVien.setColumnWidth(2, 220)
        self.tbl_SinhVien.setRowCount(len(self.lstSV))
        for sv in self.lstSV:
            self.tbl_SinhVien.setItem(row, 0, QtWidgets.QTableWidgetItem(sv["ID"]))
            self.tbl_SinhVien.setItem(row, 1, QtWidgets.QTableWidgetItem(sv["name"]))
            self.tbl_SinhVien.setItem(row, 2, QtWidgets.QTableWidgetItem(sv["class"]))
            row += 1

if __name__=="__main__":
    app=QApplication(sys.argv)
    ui=MAIN()
    app.exec_()