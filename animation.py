import sys
import random
from PySide6 import QtCore
from PySide6.QtWidgets import *
from PySide6.QtGui import *
import json

states = [[0,1,2],[2,1,2]]
actions = [[0,1] ]
class Neuron:
    def __init__(self, id, coordinates,value):
        self.id = id
        self.coordinates = coordinates
        self.value = value
        self.color = QtCore.Qt.black
    
    def get_coordinates(self):
        return self.coordinates[0], self.coordinates[1]

class Net:
    def __init__(self, nb_inputs,nb_outputs,  coordinates, width, height, neuron_radius, rotate = False):
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        
        self.coordinates = coordinates
        self.width = width
        self.height = height
        self.neuron_radius = neuron_radius
        self.input_neurons = []
        self.input_neuron_labels = []
        self.output_neurons = []
        self.rotate = rotate
        self.build_network()
        self.__init_neuron_labels()
    
    def build_network(self):
        if self.rotate:
            start_x = self.coordinates[0] + (self.width/2)
            if self.nb_inputs >= self.nb_inputs:
                self.neuron_offset = self.height / self.nb_inputs
                inputs_start_y = self.coordinates[1] - (self.height/2)
                outputs_start_y = self.coordinates[1] - ((self.nb_outputs * self.neuron_offset) / 2)
            else:
                self.neuron_offset = self.height / self.nb_outputs
                outputs_start_y = self.coordinates[1] - (self.height/2)
                inputs_start_y = self.coordinates[1] - ((self.nb_inputs * self.neuron_offset) / 2)
                
            for i in range(0,self.nb_inputs):
                self.input_neurons.append(Neuron(i,(start_x,inputs_start_y+(i*self.neuron_offset)),0))
            
            for i in range(0,self.nb_outputs):
                self.output_neurons.append(Neuron(i,(start_x-self.width, outputs_start_y+(i*self.neuron_offset)),0))    
        
        else:
            if self.nb_inputs >= self.nb_inputs:
                self.neuron_offset = self.width / self.nb_inputs
                inputs_start_x = self.coordinates[0] - (self.width/2)
                outputs_start_x = self.coordinates[0] - ((self.nb_outputs * self.neuron_offset) / 2)
            
            else:
                self.neuron_offset = self.width / self.nb_outputs
                outputs_start_x = self.coordinates[0] - (self.width/2)
                inputs_start_x = self.coordinates[0] - ((self.nb_inputs * self.neuron_offset) / 2)
        
            
            start_y = self.coordinates[1] + (self.height/2)
            for i in range(0,self.nb_inputs):
                self.input_neurons.append(Neuron(i,(inputs_start_x+(i*self.neuron_offset),start_y),0))
            
            for i in range(0,self.nb_outputs):
                self.output_neurons.append(Neuron(i,(outputs_start_x+(i*self.neuron_offset),start_y-self.height),0))
    
    def __init_neuron_labels(self):
        pass    
            #if self.rotate:
                

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.hello = ["Hallo Welt", "Hei maailma", "Hola Mundo", "Привет мир"]

        self.button = QPushButton("Click me!")
        self.text = QLabel("Hello World",
                                     alignment=QtCore.Qt.AlignCenter)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.button)

        self.button.clicked.connect(self.magic)

    @QtCore.Slot()
    def magic(self):
        self.text.setText(random.choice(self.hello))

class Controler(QMainWindow):
    def __init__(self, data,net1, net2):
        super().__init__()
        
        self.setGeometry(100,100,1440,1000)
        
        #central_widget = QWidget()
        #self.setCentralWidget(central_widget)
        #self.layout = QVBoxLayout(central_widget) # Použijeme vertikálny layout
        self.steps_data = data
        self.nb_steps = 7
        self.phase_map = [
            "Update state",
            "Run inference",
            "Find max Q value of next possible states",
            "Compute Q value of current state",
            "Get reward",
            "Set target",
            "Update weights",
        ]
        self.last_step_next = False
        
        self.input_labels = []
        self.output_labels = []
        self.equation_label = None
        self.equation2_label = None
        self.step_viewer = None
        self.prev_button = 0
        self.next_button = 0
        self.animation_counter = 0
        self.step_counter = 0
        self.net1 = net1
        self.net2 = net2
        self.font_size = 14
        self.flesh_time = 1500
        self.reward  = 0
        self.gamma = 0.99
        self.Q_new = 0
        self.Q_max = 0
        
        # --- Vytvorenie labelov (LEN RAZ) ---
        self.net1_input_l,self.net1_output_l  = self.create_labels(net1, data[0]["outputs_Q"], data[0]["inputs"])
        self.net2_input_l,self.net2_output_l  = self.create_labels(net2, data[0]["outputs_Q"], data[0]["inputs"])
        
        self.init_Equations(450,600)
        self.create_buttons(100,0)
        self.create_step_viewer(300,0)
        self.show()
        
    def paintEvent(self, event):
        qp=QPainter()
        qp.begin(self)
        self.draw_neurons(qp, net1)
        self.draw_neurons(qp, net2)
        qp.end
    
    def draw_neurons(self,qp,net):
        for neuron in net.input_neurons:
            c = neuron.get_coordinates()
            pen = QPen(neuron.color,2,QtCore.Qt.SolidLine)
            qp.setPen(pen)
        
            qp.drawEllipse(c[0]-(net.neuron_radius), c[1]-(net.neuron_radius),net.neuron_radius*2,net.neuron_radius*2)
        
        for neuron in net.output_neurons:
            c = neuron.get_coordinates()
            pen = QPen(neuron.color,2,QtCore.Qt.SolidLine)
            qp.setPen(pen)
        
            qp.drawEllipse(c[0]-(net.neuron_radius), c[1]-(net.neuron_radius),net.neuron_radius*2,net.neuron_radius*2)
    
    def flash_neurons(self, net):
        for n in net.input_neurons:
            n.color = QtCore.Qt.green
        for n in net.output_neurons:
            n.color = QtCore.Qt.green 
        
        self.update()
        QtCore.QTimer.singleShot(self.flesh_time, lambda: self.reset_neurons_color(net))
    
        
    def reset_neurons_color(self, net):
        for n in net.input_neurons:
            n.color = QtCore.Qt.black
        for n in net.output_neurons:
            n.color = QtCore.Qt.black
        self.update()
        
    def create_labels(self, net, input_dict, output_dict):
        input_labels = []
        output_labels = []
        for neuron,(k,_) in zip(net.input_neurons,input_dict.items()):
            (c_x,c_y) = neuron.coordinates
            
            label = QLabel(f"{k}:\n0",self)
            label.setStyleSheet("color: black; font-size: {self.font_size}pt;")
            label.setAlignment(Qt.AlignCenter)
            label.setGeometry(c_x+50,c_y-25,150,50)
            output_labels.append(label)
            
            
        for neuron,(k,_) in zip(net.output_neurons,output_dict.items()):
            (c_x,c_y) = neuron.coordinates
            
            label = QLabel(f"{k}:\n0",self)
            label.setStyleSheet("color: black; font-size: {self.font_size}pt;")
            label.setAlignment(Qt.AlignCenter)
            label.setGeometry(c_x-150,c_y-25,150,50)
            input_labels.append(label)
        
        
        return input_labels, output_labels
            
    def update_inputs(self, input_dict, input_labels):
        for l,(k,v) in zip(input_labels, input_dict.items()):
            l.setText(f"{k}:\n{v:.3f}")
            l.setStyleSheet("color: blue; font-size: {self.font_size}pt;")
            
        QtCore.QTimer.singleShot(self.flesh_time, lambda: self.reset_label_color(input_labels))
    
    def update_Q_labels(self, output_dict, output_labels):
        for l,(k,v) in zip(output_labels, output_dict.items()):
            l.setText(f"{k}:\n{v:.3f}")
            l.setStyleSheet("color: blue; font-size: {self.font_size}pt;")
        
        QtCore.QTimer.singleShot(self.flesh_time, lambda: self.reset_label_color(output_labels))
    
    def reset_label_color(self, labels):
        for i,l in enumerate(labels):
            l.setStyleSheet("color: black; font-size: {self.font_size}pt;")
    
    def flash_Q_label(self,labels, Q_id):
        labels[Q_id].setStyleSheet("color: red; font-size: {self.font_size}pt;")
        QtCore.QTimer.singleShot(self.flesh_time, lambda: self.reset_label_color(labels))
    
    
        
    #def update_target_labels(self, step):
    #    for i,l in enumerate(self.output_labels_labels):
    #        l.setText(list(step["outputs_target"])[i])
    #        l.setStyleSheet("color: blue; font-size: 20pt;")
    
    def init_Equations(self, x,y):
        equation_html = """
        <div style="text-align: center; font-family: monospace; font-size: 36px; padding: 20px;">
            <p>Q<sub>s,a_i</sub> = R + &gamma; &sdot; max(Q<sub>s_next,a</sub>)</p>
        </div>
        """
        
        self.equation_label = QLabel("",self)
        self.equation_label.setText(equation_html)
        self.equation_label.setGeometry(x,y,800,600)
        
        equation2_html = f"""
        <div style="text-align: center; font-family: monospace; font-size: 36px; padding: 20px;">
            <p>{self.Q_new}</sub> = {self.reward} + {self.gamma} &sdot; {self.Q_max}</p>
        </div>
        """
        
        self.equation2_label = QLabel("",self)
        self.equation2_label.setText(equation2_html)
        #equation_label.setAlignment(Qt.AlignCenter)
        self.equation2_label.setGeometry(x,y+80,800,600)
        
    def update_eq(self, flash_term):
        Q_n_color = "black"
        Q_m_color = "black"
        r_color = "balck"
        if flash_term == "Q_new":
            Q_n_color = "red"
        elif flash_term == "reward":
            r_color = "red"
        elif flash_term == "Q_max":
            Q_m_color = "red"
        
        equation_html = f"""
        <div style="text-align: center; font-family: monospace; font-size: 36px; padding: 20px;">
            <p><span style="color: {Q_n_color};">Q<sub>s,a_i</sub></span> = <span style="color: {r_color};">R</span> + &gamma; &sdot; <span style="color: {Q_m_color};">max(Q<sub>s_next,a</sub>)</span></p>
        </div>
        """
        self.equation_label.setText(equation_html)
        
        #self.reward = reward
        equation2_html = f"""
        <div style="text-align: center; font-family: monospace; font-size: 36px; padding: 20px;">
            <p><span style="color: {Q_n_color};">{self.Q_new:.3f}</sub></span> = <span style="color: {r_color};">{self.reward:.3f}</span> + {self.gamma:.3f} &sdot; <span style="color: {Q_m_color};">{self.Q_max:.3f}</span></p>
        </div>"""
        self.equation2_label.setText(equation2_html)
        
        QtCore.QTimer.singleShot(self.flesh_time, self.reset_eq_reward_color)
    
        pass
    def reset_eq_reward_color(self):
        equation_html = """
        <div style="text-align: center; font-family: monospace; font-size: 36px; padding: 20px;">
            <p>Q<sub>s,a_i</sub> = R + &gamma; &sdot; max(Q<sub>s_next,a</sub>)</p>
        </div>
        """
        self.equation_label.setText(equation_html)
        
        equation2_html = f"""
        <div style="text-align: center; font-family: monospace; font-size: 36px; padding: 20px;">
            <p>{self.Q_new:.3f} =  {self.reward:.3f} + {self.gamma:.3f} &sdot; {self.Q_max:.3f}</p>
        </div>"""
        self.equation2_label.setText(equation2_html)
    
    def create_buttons(self,x,y):
        self.prev_button = QPushButton("<< Previous", self)
        self.next_button = QPushButton("Next >>", self)
        
        self.next_button.setGeometry(x,y,100,30)
        self.prev_button.setGeometry(x-100,y,100,30)
        
        self._connect_signals()
    
    def create_step_viewer(self, x, y):
        self.step_viewer = QLabel(f"Init",self)
        self.step_viewer.setStyleSheet("font-size: 20pt;")
        self.step_viewer.setGeometry(x,y,800,40)
    
    def update_step_viewer(self):
        self.step_viewer.setText(f"step:{self.step_counter+1} phase: {self.phase_map[self.animation_counter%self.nb_steps]}")
        
    def _connect_signals(self):
        self.prev_button.clicked.connect(self._previous_frame)
        self.next_button.clicked.connect(self._next_frame)

    # --- Logika Controlleru: Riadenie Indexu ---

    
    def _previous_frame(self):
        
        
        # counter was already increased if last step was next so it needs to be decreased twice
        if self.last_step_next:
            self.animation_counter-=1
            self.last_step_next = False
            
        self.animation_counter-=1
        
        # edge conditions
        if self.animation_counter < 0:
            print("hi")
            self.animation_counter=0
            return None
        
        self.update_view()

    def _next_frame(self):
        self.last_step_next = True
        self.update_view()
        self.animation_counter+=1
        
    
    def update_view(self):
        self.update_step_viewer()
        print(self.animation_counter)
        
        #if self.step_counter+1 % 10 == 0:
        #    self.flash_neurons(net1)
        #    self.flash_neurons(net2)
        #    return None
            
        if self.animation_counter % self.nb_steps==0:    
            self.update_inputs(self.steps_data[self.step_counter]["inputs"], self.net1_input_l)
            self.update_inputs(self.steps_data[self.step_counter]["inputs"], self.net2_input_l)
        elif self.animation_counter % self.nb_steps==1:
            self.update_Q_labels(self.steps_data[self.step_counter]["outputs_Q"], self.net1_output_l)
            self.update_Q_labels(self.steps_data[self.step_counter]["outputs_target"], self.net2_output_l)
        elif self.animation_counter % self.nb_steps==2:
            action_id = self.steps_data[self.step_counter]["max_Q_id"]
            self.flash_Q_label(self.net1_output_l, action_id)
            self.Q_max = list(self.steps_data[self.step_counter]["outputs_Q"].values())[action_id]
            self.update_eq("Q_max")
        elif self.animation_counter % self.nb_steps==3:
            self.Q_new = self.steps_data[self.step_counter]["Q"]
            self.update_eq("Q_new")
        elif self.animation_counter % self.nb_steps==4:
            self.update_eq("reward")
        elif self.animation_counter % self.nb_steps==5:
            self.reward = self.steps_data[self.step_counter]["reward"]
        elif self.animation_counter % self.nb_steps==6:
            self.flash_neurons(net1)
            self.step_counter+=1
        
        
    
        
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    json_file = sys.argv[1]
    
    with open(json_file) as f:
        data = json.load(f)
        
    nb_inputs = len(data[0]["inputs"])
    nb_outputs = len(data[0]["outputs_Q"])
    
    net1 = Net(nb_outputs,nb_inputs,(350,480),200,700,25, True)
    net2 = Net(nb_outputs,nb_inputs,(950,480),200,700,25, True)
    
    contorler = Controler(data,net1,net2)
    # widget = MyWidget() 
    # widget.resize(800, 600)
    # widget.show()

    sys.exit(app.exec())