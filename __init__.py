from dashing import *
from time import sleep, time
import math, PIL.Image 
from PIL import Image
from blessed import Terminal

from Net import Net


if __name__ == "__main__":    
    net = Net()

    t0 = 'Layer 0    Layer 1     Layer 2                     |  Update type: {0}\n'.format(net.update_type)
    t1 = '------     -----                                   |  Learning Rate: {0}\n'.format(net.learning_rate)
    t2 = '|_0_|------|_0_|- -                                |  Momentum: {0}\n'.format(net.momentum)
    t4 = '----- - -  -----    -  -----     Expected Output   ------------------------------------------------\n'
    t5 = '|_1_|------|_1_|-------|_0_|-------> 0 or 1        \n'
    t7 = '-----  - - -----  -  - -----                       \n'
    t8 = '|_B_|------|_2_|-   -  |_B_|                       \n'
    t10 = '       - - ----- - -                              \n'
    t11 = '       -  -|_3_|- -                               \n'
    t12 = '        -  ----- -                                \n'
    t13 = '          -|_B_|-                                 \n'
    display = t0 + t1 + t2 + t4 + t5 + t7 + t8 + t10 + t11 + t12 + t13

    term = Terminal()
    ui = HSplit(
            VSplit(
                Text(text=f"{display}", title='Network Display', border_color=2, color=3),
                Log(title='Training Data', border_color=2, color=3),
                Log(title='|---EPOCH--|----------------MSE------------|--------------OUTPUT--------------|------EXPECTED-------|', border_color=5, color=3),
                HGauge(val=50, title="Epochs", border_color=5),
            ),      
            VSplit(
                Log(title='Layer 0 to Layer 1 (Delta, Batch Delta Weight)', border_color=2, color=3),
                Log(title='Layer 1 to Layer 2 (Delta, Batch Delta Weight)', border_color=2, color=3),
            ),                    
        )
    data = ui.items[0].items[2]
    train_data = ui.items[0].items[1]

    l0l1_log = ui.items[1].items[0]
    l1l2_log = ui.items[1].items[1]

    val = ''

    with term.cbreak():
        while val.lower() != 'q':
            val = term.inkey(timeout=0.15)
            if not val:                
                net.run_epoch()
                data.append(f"     {net.epoch}    |        {net.MSE}    |            {net.output_}    |           {net.expected_value}")
                train_data.append(f"{net.training_data.next_line}")
                l0l1_log.append(f"{net.delta_printL0L1()}")
                l1l2_log.append(f"{net.delta_printL1L2()}")
                ui.items[0].items[3].value = int(100 * math.sin(net.epoch / 100.0))
                ui.display()
            
            if ui.items[0].items[3].value == 99:
                print('Max epoch reached')
                break
                        
    print('Bye')        
    sleep(1.0/25)

    # ui = HSplit(
    #         VSplit(
    #             HGauge(val=50, title="only title", border_color=5),
    #             Log(title='Logs', border_color=5, color=3)
    #         ),
    #         VSplit(
    #             Text('Hello World,\nthis is dashing.', border_color=2),
    #             Log(title='logs', border_color=5),
    #             VChart(border_color=2, color=2),
    #             HChart(border_color=2, color=2),
    #             HBrailleChart(border_color=2, color=2),
    #         ),
    #         title='Dashing',
    #     )
    # val = ''
    # count = 0
    # log = ui.items[0].items[1]
    # vchart = ui.items[1].items[2]
    # hchart = ui.items[1].items[3]
    # bchart = ui.items[1].items[4]

    # with term.cbreak():
    #     while val.lower() != 'q': 
    #         val = term.inkey(timeout=0.1)
    #         net.run_epoch()       
    #         count += 1
    #         log.append(str(net.epoch))

    #         prev_time = time()
    #         t = int(time())
    #         if t != prev_time:                
    #             ui.items[0].items[0].value = int(50 * math.sin(count / 80.0))
    #             ui.display()

  