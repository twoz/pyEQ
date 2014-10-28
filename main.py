from PySide.QtCore import*
from PySide.QtGui import*
import pyaudio
import wave
import time
import numpy as np
from collections import OrderedDict
from filters import FilterType, Filter, FilterChain
from utility import byteToPCM, floatToPCM, pcmToFloat, sosfreqz, toPixelCords, fromPixelCords

filterTypes = OrderedDict({
    FilterType.LPButter: 'Low Pass (Flat)', 
    FilterType.LPBrickwall: 'Low Pass (Brickwall)',
    FilterType.HPButter: 'High Pass (Flat)',
    FilterType.HPBrickwall: 'High Pass (Brickwall)',
    FilterType.LShelving: 'Low Shelf',
    FilterType.HShelving: 'High Shelf',
    FilterType.Peak: 'Peak'})


fs = 44100
eps = 0.0000001

class Params:
    TYPE = 1
    F = 2
    G = 3
    Q = 4

class NodeLayout(QGridLayout):

    updated = Signal(int, int, int) #(node index, parameter, new value)
    enabled = Signal(int) #node index

    def __init__(self, index, mainwin, *args):
        QGridLayout.__init__(self, *args)       
        self.index = index
        self.mainwin = mainwin

    def addControls(self, checkbox, ftype_combo, f_ledit, gain_ledit, q_slider):

        self.ctrls = [checkbox, ftype_combo, f_ledit, gain_ledit, q_slider]
        
        ftype_combo.currentIndexChanged.connect(self.typeChanged)
        self.addWidget(ftype_combo, 0, 0)

        self.addWidget(checkbox, 0, 1)
        checkbox.clicked.connect(self.nodeStateChanged)

        f_ledit.editingFinished.connect(self.freqChanged)
        self.addWidget(f_ledit, 1, 0)
        self.addWidget(QLabel('Hz'), 1, 1)

        gain_ledit.editingFinished.connect(self.gainChanged)
        self.addWidget(gain_ledit, 2, 0)
        self.addWidget(QLabel('dB'), 2, 1)
        
        q_slider.sliderMoved.connect(self.qSliderMoved)
        self.addWidget(q_slider, 3, 0)
        self.slider_label = QLabel('')
        self.addWidget(self.slider_label, 3, 1)
    
    def isEnabled(self):
        return self.ctrls[0].isChecked()

    def setControlsEnabled(self, enabled):
        for i in range(1,len(self.ctrls)):
            self.ctrls[i].setEnabled(enabled)

    def setControlEnabled(self, index, enabled):
        self.ctrls[index].setEnabled(enabled)

    def nodeStateChanged(self):
        self.enabled.emit(self.index)

    def typeChanged(self, index):
        self.updated.emit(self.index, Params.TYPE, index)

    def freqChanged(self):
        self.updated.emit(self.index, Params.F, 0)

    def gainChanged(self):
        self.updated.emit(self.index, Params.G, 0)

    def qSliderMoved(self, val):
        self.updated.emit(self.index, Params.Q, val)

class Axis(object):
    def __init__(self, type, min, max, log = False, *args):
        self.type = type
        self.min = min
        self.max = max
        self.log = log

class PlotCurve(object):
    def __init__(self, pen, brush = QBrush(), is_path = False, *args):
        self.is_path = is_path
        self.pen = pen
        self.brush = brush
        self.xdata = []
        self.ydata = []

    def setData(self, x, y):
        if self.is_path:
            self._path = QPainterPath()

        self.xdata = x
        self.ydata = y

class PlotWin(QFrame):
    def __init__(self, *args):
        QFrame.__init__(self, *args)
        self.setFrameStyle(QFrame.Box)
        self.setAutoFillBackground(True)
        self.show()

        self.xaxis = Axis('bottom', 50, fs / 2, log = True)
        self.laxis = Axis('left', -100, 0)
        self.raxis = Axis('right', -10, 10)
        self.rect = QRectF()

        pen1 = QPen(Qt.gray)
        pen1.setWidth(1.5)
        self.speccurv = PlotCurve(pen1)
        pen2 = QPen()
        pen2.setWidth(2)
        pen2.setColor(Qt.red)
        self.TFcurv = PlotCurve(pen2)
        w0 = self.xaxis.min * 2 * np.pi / fs
        self.wor = np.logspace(np.log10(w0), np.log10(np.pi), 512)
        self.refresh_rate = 30

        self.chain = None
        self.handles = [QPoint()] * 5
        self.dragged = False
        self.focused = -1

    def resizeEvent(self, e):
        QFrame.resizeEvent(self, e)
        gradient = QLinearGradient(QPointF(self.width() / 2, 0), QPointF(self.width() / 2, self.height()))
        gradient.setColorAt(0, QColor(100, 102, 127))
        gradient.setColorAt(1, QColor(0, 0, 0))
        p = self.palette()
        p.setBrush(QPalette.Background, QBrush(gradient))        
        self.setPalette(p)

    def mousePressEvent(self, e):
        QFrame.mousePressEvent(self, e)

        for i, h in enumerate(self.handles):
            if h != None:
                if abs(e.pos().x() - h.x()) <= 10 and abs(e.pos().y() - h.y()) <= 10:
                    self.dragged = True
                    self.focused = i
                    self.cursor().setPos(self.mapToGlobal(h))
                    QApplication.setOverrideCursor(Qt.BlankCursor)
                    self.update()

    def mouseMoveEvent(self, e):
        QFrame.mouseMoveEvent(self, e)
        if self.dragged:
            pos, i = e.pos(), self.focused
            fc, g = fromPixelCords(self.width(), self.height(), pos, self.xaxis, self.raxis)
            old = self.parent().chain._filters[i]
            if old._type not in (FilterType.Peak, FilterType.LShelving, FilterType.HShelving):
                g = 0
            self.parent().chain.updateFilt(i,Filter(old._type, fc * 2 / fs, g, Q = old._Q))
            self.updateHandles()
            self.parent().updateChainTF()
            self.update()
             
    def mouseReleaseEvent(self, e):
        QFrame.mouseReleaseEvent(self, e)
        self.dragged = False
        QApplication.restoreOverrideCursor()

    def paintEvent(self, e):
        QFrame.paintEvent(self, e)
        self.rect = QRectF(0, 0, self.width(), self.height() - 10)

        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)

        self.drawTicks(qp, self.xaxis)
        self.drawTicks(qp, self.laxis)
        self.drawTicks(qp, self.raxis)      

        #paint filter response
        filt = self.parent().chain._filters[self.focused]
        if filt._enabled:
            w, H = sosfreqz(filt._sos, self.wor)
            pen = QPen((QColor(170, 0, 0)))
            pen.setWidth(1.5)
            if filt._type == FilterType.Peak:
                c = PlotCurve(pen, QBrush(QColor(255, 255, 255, 50)), is_path = True)
            else:
                c = PlotCurve(pen, is_path = True)

            c.setData(w * 0.5 / np.pi * fs, 20 * np.log10(np.abs(H) + eps))
            self.plot(qp, c, self.raxis)

        #paint chain response
        self.plot(qp, self.TFcurv, self.raxis)
         
        #paint handles      
        self.drawHandles(qp)  
        
        #paint the spectrum
        self.plot(qp, self.speccurv, self.laxis)    

    def plot(self, qp, curve, yaxis):
        w = self.width()
        h = self.height()
        qp.setPen(curve.pen)
        qp.setBrush(curve.brush)
        
        if curve.is_path:
            if curve._path.elementCount() == 0:                
                curve._path.moveTo(toPixelCords(w, h, 0, self.xaxis, 0, yaxis))
                for x, y in zip(curve.xdata, curve.ydata):
                    curve._path.lineTo(toPixelCords(w, h, x, self.xaxis, y, yaxis))
            qp.drawPath(curve._path)
        else:
            poly = QPolygon()
            for x, y in zip(curve.xdata, curve.ydata):
                poly << toPixelCords(w, h, x, self.xaxis, y, yaxis)
            qp.drawPolyline(poly)

    def updateHandles(self):

        for i, filter in enumerate(self.parent().chain._filters):
            if filter._enabled is True:
                fc = filter._fc * fs * 0.5
                if filter._type not in (FilterType.Peak, FilterType.LShelving, FilterType.HShelving):
                    y = 0
                else:
                    y = filter._g

                self.handles[i] = toPixelCords(self.width(), self.height(), fc, self.xaxis, y, self.raxis)
                self.parent().nodes[i].ctrls[2].setText(str(int(fc)))
                self.parent().nodes[i].ctrls[3].setText("{:.1f}".format(filter._g))
            else:
                self.handles[i] = None

    def updateSpectrum(self, dft):

        if dft.size > 0:
                    N = dft.size
                    self.speccurv.setData([fs / 2 / N * i for i in range(0,N)],
                                                   20 * np.log10(np.abs(dft / N) + eps))
                    self.update()

    def drawHandles(self, qp):
                         
        for i, h in enumerate(self.handles): 
            if h != None:
                if self.focused == i:
                    alpha = 255
                    m = 1
                else:
                    alpha = 90
                    m = 1.3
                pen = QPen(QColor(255,255,255,alpha))
                pen.setWidth(2)
                qp.setPen(pen)  
                qp.setBrush(QBrush(QColor(150,150,150,alpha)))
                qp.drawEllipse(h, 12 / m, 12 / m)
                qp.setBrush(QBrush(QColor(0,0,0,alpha)))
                qp.drawEllipse(h, 4 / m, 4 / m)
        
    def drawTicks(self, qp, axis):
        tick_pen = QPen(QColor(200, 200, 200))
        qp.setPen(tick_pen)
        grid_major_pen = QPen(QColor(255, 255, 255, 80))
        grid_major_pen.setStyle(Qt.DashLine)
        grid_minor_pen = QPen(QColor(255, 255, 255, 40))
        grid_minor_pen.setStyle(Qt.DashLine)
        majors = []
        minors = []
        ticklen = 10
        w = self.width()
        h = self.height()
        xaxis = self.xaxis
        bgap = self.height() - self.rect.height()

        
        if axis.type == 'bottom':
            majors = [100, 1000, 10000]
            for i in range(1,5):
                minors.extend([j * 10 ** i for j in range(2,10)])

            qp.drawLine(self.rect.bottomLeft(), self.rect.bottomRight())          
            for tick in majors:

                qp.setPen(tick_pen)
                qp.drawLine(toPixelCords(w, h, tick, xaxis), h - bgap, toPixelCords(w, h, tick, xaxis), h - bgap - ticklen)
                qp.drawText(toPixelCords(w, h, tick, xaxis) - bgap, h, str(tick))

                qp.setPen(grid_major_pen)
                qp.drawLine(toPixelCords(w, h, tick, xaxis), h - bgap - ticklen, toPixelCords(w, h, tick, xaxis), 0)

            for tick in minors:
                qp.setPen(tick_pen)
                qp.drawLine(toPixelCords(w, h, tick, xaxis), h - bgap, toPixelCords(w, h, tick, xaxis), h - bgap - ticklen * 0.5)
              
                qp.setPen(grid_minor_pen)
                qp.drawLine(toPixelCords(w, h, tick, xaxis), h - bgap - ticklen * 0.5, toPixelCords(w, h, tick, xaxis), 0)

        elif axis.type == 'left':

            i = 1
            while -i * 10 > axis.min:
                majors.append(-i * 10)
                i = i + 1
            qp.drawLine(0, 0, 0, self.height() - bgap)
            qp.drawText(ticklen, 15, '[dB]')
            for tick in majors:
                
                qp.setPen(tick_pen)
                yp = toPixelCords(w, h, 0, xaxis, tick, self.laxis).y()
                qp.drawLine(0, yp, ticklen * 0.5, yp)
                qp.drawText(ticklen, yp + 4, str(tick))

                qp.setPen(grid_major_pen)
                qp.drawLine(ticklen * 0.5, yp, self.width(), yp)

        elif axis.type == 'right':
            n = 11
            majors = np.linspace(axis.min, axis.max, n)
            qp.drawLine(self.rect.topRight(), self.rect.bottomRight())
            for tick in majors:
                yp = toPixelCords(w, h, 0, xaxis, tick, axis).y()
                qp.drawLine(self.width() - ticklen * 0.5, yp,
                            self.width(), yp)
                qp.drawText(self.width() - ticklen - 10, yp + 4, str(int(tick)))
       
class MainWindow(QWidget):
    def __init__(self, *args):
        QWidget.__init__(self, *args)

        self.setFixedSize(1000,500)
        self.setWindowTitle('EQ')
        self.show()

        layout = QVBoxLayout(self)

        #--------- track controls -----------
        open_btn = QPushButton('Load file')
        open_btn.clicked.connect(self.onOpenBtnClick)
        self.path_label = QLabel('')
        self.loop_box = QCheckBox('Loop')
        play_btn = QPushButton('Play')
        play_btn.clicked.connect(self.onPlayBtnClick)
        stop_btn = QPushButton('Stop')
        stop_btn.clicked.connect(self.onStopBtnClick)
        save_btn = QPushButton('Apply EQ and save')
        save_btn.clicked.connect(self.onSaveBtnClick)

        trackctrl_layout = QHBoxLayout()
        trackctrl_layout.addWidget(open_btn)
        trackctrl_layout.addWidget(self.path_label)       
        trackctrl_layout.addWidget(play_btn)
        trackctrl_layout.addWidget(stop_btn)
        trackctrl_layout.addWidget(self.loop_box)
        trackctrl_layout.addSpacing(50)
        trackctrl_layout.addWidget(save_btn)        
        layout.addLayout(trackctrl_layout)

        #--------- plot ------------
        
        self.plotwin = PlotWin(self)
        layout.addWidget(self.plotwin)

        #--------- filter controls ----------
        sub_layout = QHBoxLayout()
        labels_layout = QVBoxLayout()
        labels_layout.addWidget(QLabel('Filter type'))
        labels_layout.addWidget(QLabel('Cutoff/Center'))
        labels_layout.addWidget(QLabel('Gain'))
        labels_layout.addWidget(QLabel('Q or Slope'))
        sub_layout.addLayout(labels_layout)

        self.nodes = []
        deffs = [100, 1000, 3000, 5000, 15000]
        for i in range(0,5):
            filter_list = QComboBox()
            filter_list.addItems(list(filterTypes.values()))
            if i == 0:
                filter_list.setCurrentIndex(FilterType.HPBrickwall)
            elif i == 4:
                filter_list.setCurrentIndex(FilterType.LPBrickwall)
            else:
                filter_list.setCurrentIndex(FilterType.Peak)

            checkbox = QCheckBox('On')
            freq_txt = QLineEdit(str(deffs[i]))
            freq_txt.setValidator(QIntValidator(self.plotwin.xaxis.min,
                                               self.plotwin.xaxis.max, self))
            gain_txt = QLineEdit('0')
            gain_txt.setValidator(QDoubleValidator(-12, 12, 1, self))
            q_slider = QSlider(Qt.Horizontal)
            node = NodeLayout(i, self)
            node.addControls(checkbox, filter_list, freq_txt, gain_txt, q_slider)
            node.setControlsEnabled(False)
            self.nodes.append(node)
            sub_layout.addLayout(node)
            
            node.enabled.connect(self.onFilterEnableChange)
            node.updated.connect(self.paramChanged)

        layout.addLayout(sub_layout)
        #------------------------------------
        self.setLayout(layout)

        #----------- Filters ----------------
        self.chain = FilterChain()
        deffs = [fc * 2 / fs for fc in deffs]
        self.chain._filters.append(Filter(FilterType.HPBrickwall, deffs[0], enabled = False))
        self.chain._filters.append(Filter(FilterType.Peak, deffs[1], enabled = False))
        self.chain._filters.append(Filter(FilterType.Peak, deffs[2], enabled = False))
        self.chain._filters.append(Filter(FilterType.Peak, deffs[3], enabled = False))
        self.chain._filters.append(Filter(FilterType.LPBrickwall, deffs[4], enabled = False))
        self.updateChainTF()
        self.plotwin.updateHandles()

        self.stream = None
        self.wf = None

    @Slot()
    def onOpenBtnClick(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter('Audio (*.wav)')
        if dialog.exec_():
            file_name = dialog.selectedFiles()[0]
            self.path_label.setText(file_name)
            self.wf = wave.open(file_name,'rb')
            self.openStream()

    @Slot()
    def onPlayBtnClick(self):
        if self.stream:
            if self.wf.tell() == self.wf.getnframes() * self.wf.getnchannels():
                self.wf.rewind()
                self.openStream()
            else:
                self.stream.start_stream()


    @Slot()
    def onStopBtnClick(self):
        if self.stream:
            self.stream.stop_stream()

    @Slot()
    def onSaveBtnClick(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setNameFilter('Audio (*.wav)')
        if dialog.exec_():
            file_name = dialog.selectedFiles()[0] + '.wav'
            ww = wave.open(file_name,'wb')
            wf = self.wf
            ww.setframerate(wf.getframerate())
            ww.setsampwidth(wf.getsampwidth())
            ww.setnchannels(wf.getnchannels())
            self.wf.rewind()
            self.chain.reset()

            data = wf.readframes(wf.getnframes())
            s = self.chain.filter(pcmToFloat(byteToPCM(data,wf.getsampwidth())))
            ww.writeframes(bytes(floatToPCM(s)))

    @Slot()
    def onFilterEnableChange(self, i):        
        enabled = self.nodes[i].ctrls[0].isChecked()
        if enabled:
            ftype = self.nodes[i].ctrls[1].currentIndex()
            self.updateControls(i, ftype)
            self.adjustSliderRange(i, ftype)
            self.updateSliderLabel(i)
        else:
            self.nodes[i].setControlsEnabled(False)
        
        self.chain.setFiltEnabled(i, enabled)
        self.plotwin.updateHandles() 
        self.updateChainTF()

    @Slot()
    def paramChanged(self, i, param, val):
        self.updateFilter(i, param, val)       
        self.updateChainTF()
        self.plotwin.updateHandles() 

    @Slot()
    def focusChanged(self, old, new):
        if new is not None:
            for node in self.nodes:
                if node.indexOf(new) != -1:
                    self.plotwin.focused = node.index
                    self.plotwin.update()
           
    def updateControls(self, i, ftype):
        node = self.nodes[i]
        node.setControlsEnabled(True)
        if ftype == FilterType.LPBrickwall or ftype == FilterType.HPBrickwall:
            node.setControlEnabled(3,False)
            node.setControlEnabled(4,False)
        elif ftype == FilterType.LPButter or ftype == FilterType.HPButter:
            node.setControlEnabled(3,False)

        
    def updateFilter(self, i, param, val):
        oldf = self.chain._filters[i]
        type = oldf._type
        fc = oldf._fc
        g = oldf._g
        Q = oldf._Q

        if param == Params.TYPE:
            type = val
            Q = 1                      
        elif param == Params.F:
            fc = int(self.nodes[i].ctrls[2].text()) * 2 / fs
        elif param == Params.G:
            g = float(self.nodes[i].ctrls[3].text())
        elif param == Params.Q:
            if type == FilterType.LPButter or type == FilterType.HPButter:
                Q = val
            elif type == FilterType.Peak:
                Q = val / 10
            elif type == FilterType.LShelving or FilterType.HShelving:
                Q = val / 100

        self.chain.updateFilt(i, Filter(type, fc, g, Q))
        if param == Params.TYPE:            
            self.updateControls(i, type)
            self.adjustSliderRange(i, type) 

        self.updateSliderLabel(i)    

    def adjustSliderRange(self, index, type):
        slider = self.nodes[index].ctrls[4]
        if slider.isEnabled() == False:
            return 
        Q = self.chain._filters[index]._Q
        if type == FilterType.HPButter or type == FilterType.LPButter:
            slider.setRange(1, 3)
            slider.setValue(Q)
        elif type == FilterType.Peak:
            slider.setRange(1, 300)
            slider.setValue(Q * 10)
        elif type == FilterType.LShelving or type == FilterType.HShelving:
            slider.setRange(10, 100)
            slider.setValue(Q * 100)
    
    def updateSliderLabel(self, index):
        slider = self.nodes[index].ctrls[4]
        if slider.isEnabled() == False:
            return
        type = self.chain._filters[index]._type
        Q = self.chain._filters[index]._Q
        if type == FilterType.HPButter or type == FilterType.LPButter:
            text = str(2 ** Q * 6) + ' dB/oct'
        else:
            text = str(Q)

        self.nodes[index].slider_label.setText(text)
    
    def openStream(self):

        wf = self.wf
        frate = wf.getframerate()
        sampw = wf.getsampwidth()
        nchan = wf.getnchannels()
        def callback(in_data, frame_count, time_info, status):

            data = wf.readframes(frame_count)
            if type(data) == type(''):
                data = str.encode(data)            
            if len(data) < frame_count * sampw * nchan:
                if self.loop_box.isChecked():
                    wf.rewind()                    
                    data = b''.join([data,
                                     wf.readframes(frame_count - int(len(data) / (sampw * nchan)))])
                    self.chain.reset()
                elif len(data) == 0:
                    return data, pyaudio.paComplete
 
            filtered = self.chain.filter(pcmToFloat(byteToPCM(data,sampw)))
            self.plotwin.updateSpectrum(np.fft.rfft(filtered))                
                
            return bytes(floatToPCM(filtered)), pyaudio.paContinue
        
        chunk_size = np.int(frate / self.plotwin.refresh_rate)
        self.stream = pya.open(format = pya.get_format_from_width(wf.getsampwidth()),
                                    channels = wf.getnchannels(),
                                    rate = frate,
                                    frames_per_buffer = chunk_size,
                                    output = True,
                                    stream_callback = callback)

        self.chain.reset()

    def updateChainTF(self):
   
        w, H = sosfreqz(self.chain.sos(), self.plotwin.wor)
        self.plotwin.TFcurv.setData(w * 0.5 / np.pi * fs, 20 * np.log10(np.abs(H) + eps))
        self.plotwin.update()

class App(QApplication):
    def __init__(self, *args):
        QApplication.__init__(self, *args)

        win = MainWindow()
        self.focusChanged.connect(win.focusChanged)
        win.show()
        self.exec_()

pya = pyaudio.PyAudio()
app = App([])
pya.terminate()