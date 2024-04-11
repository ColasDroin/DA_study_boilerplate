# %%
import json
from matplotlib import pyplot as plt
import xtrack as xt
import numpy as np

my_beam = 'b2' # this is the beam 4 in madx sense
collider = xt.Multiline.from_json('../collider/collider_43.json')


line = collider[f'lhc{my_beam}']

# %%
tw = line.twiss()
tw.rows['ip.*'][['x','px', 'y', 'py', 'betx', 'bety']]

# %%
line.vars[f'i_wire_ip1.{my_beam}'] = 0.0 
line.vars[f'd_wire_ip1.{my_beam}'] = 0.01 

line.vars[f'i_wire_ip5.{my_beam}'] = 0.0
line.vars[f'd_wire_ip5.{my_beam}'] = 0.01


side = 'r' if my_beam == 'b2' else 'l'
sign = 1 if my_beam == 'b2' else -1
l_name_wire = [f'bbwc.t.4{side}1', f'bbwc.b.4{side}1', f'bbwc.e.4{side}5', f'bbwc.i.4{side}5']
l_name_tct = [f'tctpv.4{side}1', f'tctpv.4{side}1', f'tctph.4{side}5', f'tctph.4{side}5']
l_h_dist = sign * np.array([0., 0., 1., -1.])
l_v_dist = sign * np.array([1., -1., 0., 0.])

# Tw to get the position of the tct, but need to discard tracker afterwards to unfreeze the line
tw = line.twiss()
l_s_tct = [((tw.rows[f'{name_tct}.{my_beam}_entry'].s + tw.rows[f'{name_tct}.{my_beam}_exit'].s)/2)[0] for name_tct in l_name_tct]
line.discard_tracker()

# upper wire in IR1
line.insert_element(name=f'bbwc.t.4r1.{my_beam}',
                    element=xt.Wire(
                        L_phy=1, 
                        L_int=2,
                        current= 0.0,
                        xma=0.0, 
                        yma= 1 # very far from the beam
                        ),
                    at_s=l_s_tct[0])
# bottom wire in IR1
line.insert_element(name=f'bbwc.b.4r1.{my_beam}',
                    element=xt.Wire(
                        L_phy=1, 
                        L_int=2, 
                        current=0.0, 
                        xma=0.0, 
                        yma=-1  # very far from the beam
                        ),
                    at_s=l_s_tct[1])

# upper wire in IR5
line.insert_element(name=f'bbwc.e.4r5.{my_beam}',
                    element=xt.Wire(
                        L_phy=1, 
                        L_int=2,
                        current= 0.0,
                        xma=1, # very far from the beam
                        yma=0.0 
                        ),
                    at_s=l_s_tct[2])
# bottom wire in IR5
line.insert_element(name=f'bbwc.i.4r5.{my_beam}',
                    element=xt.Wire(
                        L_phy=1, 
                        L_int=2, 
                        current=0.0, 
                        xma=-1, # very far from the beam
                        yma=0.0  
                        ),
                    at_s=l_s_tct[3])

# %%
tw = line.twiss()
tw.rows['ip.*'][['x','px', 'y', 'py', 'betx', 'bety']]

# %%
s_ip = line.get_s_position(at_elements='ip1')

line.vars['co_y_wire_ip1.b2'] = 0
line.vars['co_x_wire_ip1.b2'] = 0
line.vars['co_y_wire_ip5.b2'] = 0
line.vars['co_x_wire_ip5.b2'] = 0        

# %%
tw = line.twiss()
tw.rows['ip.*'][['x','px', 'y', 'py', 'betx', 'bety']]

line.element_refs['bbwc.t.4r1.b2'].current = line.vars['i_wire_ip1.b2']
line.element_refs['bbwc.t.4r1.b2'].yma = line.vars['d_wire_ip1.b2'] + line.vars['co_y_wire_ip1.b2']

line.element_refs['bbwc.b.4r1.b2'].current = line.vars['i_wire_ip1.b2'] 
line.element_refs['bbwc.b.4r1.b2'].yma = -line.vars['d_wire_ip1.b2'] + line.vars['co_y_wire_ip1.b2']

line.element_refs['bbwc.e.4r5.b2'].current = line.vars['i_wire_ip5.b2']
line.element_refs['bbwc.e.4r5.b2'].xma = line.vars['d_wire_ip5.b2'] + line.vars['co_x_wire_ip5.b2']

line.element_refs['bbwc.i.4r5.b2'].current = line.vars['i_wire_ip5.b2']
line.element_refs['bbwc.i.4r5.b2'].xma = -line.vars['d_wire_ip5.b2'] + line.vars['co_x_wire_ip5.b2']

# %%
tw = line.twiss()
print(tw.qx, tw.qy)
tw.rows['ip.*'][['x','px', 'y', 'py', 'betx', 'bety']]

# %%
for ii in [2,8]:
        collider.vars[f'on_x{ii}h'] = 0.0
        collider.vars[f'on_x{ii}v'] = 0.0
        collider.vars[f'on_sep{ii}h'] = 0.0
        collider.vars[f'on_sep{ii}v'] = 0.0

for ii in [1,2,5,8]:
        print(8*'*', f'IP{ii}', 8*'*')
        if ii in [2,8]:
                print(f'on_sep{ii}h:\t ', collider.vars[f'on_sep{ii}h']._get_value())
                print(f'on_sep{ii}v:\t ', collider.vars[f'on_sep{ii}v']._get_value())
        else:
                print(f'on_x{ii}:\t\t ', collider.vars[f'on_x{ii}']._get_value())
                print(f'on_sep{ii}:\t ', collider.vars[f'on_sep{ii}']._get_value())
        print(f'on_oh{ii}:\t\t ', collider.vars[f'on_oh{ii}']._get_value())
        print(f'on_ov{ii}:\t\t ', collider.vars[f'on_ov{ii}']._get_value())
        print(f'on_a{ii}:\t\t ', collider.vars[f'on_a{ii}']._get_value())

print(8*'*', 'others settings', 8*'*')
print('on_alice_normalized:\t', collider.vars['on_alice_normalized']._get_value())
print('on_lhcb_normalized:\t', collider.vars['on_lhcb_normalized']._get_value())
print('on_disp:\t\t', collider.vars['on_disp']._get_value())

# %%
# import json file 
with open('/afs/cern.ch/work/c/cdroin/private/example_DA_study_runIII_wire/master_study/master_jobs/knobs_wire/knob_dict_350A_8sigma@30cm_ip5_beta30_b2.json') as f:
    data = json.load(f)

line.vars['d_wire_ip1.b2'] = data['tct_opening_in_sigma'] * data['sigma_y_at_tctpv_4r1_b2'] + data['wire_retraction']
line.vars['d_wire_ip5.b2'] = data['tct_opening_in_sigma'] * data['sigma_x_at_tctph_4r5_b2'] + data['wire_retraction']

# %%
tw = line.twiss()
print(tw.qx, tw.qy)
tw.rows['ip.*'][['x','px', 'y', 'py', 'betx', 'bety']]
# %%

for ii in data['k_0']:
    assert data['k_0'][ii] == line.vars[ii]._get_value()
 
# %%
my_k_list = [
           'kq5.l1b2',
           'kq5.r1b2', 
           'kq6.l1b2', 
           'kq6.r1b2',
           'kq7.l1b2',
           'kq7.r1b2',  
           'kq8.l1b2',
           'kq8.r1b2', 
           'kq9.l1b2', 
           'kq9.r1b2',
           'kq10.l1b2',
           'kq10.r1b2', 
           'kqtl11.r1b2', 
           'kqt12.r1b2', 
           'kqt13.r1b2',
           'kq4.l5b2',
           'kq4.r5b2',
           'kq5.l5b2',
           'kq5.r5b2', 
           'kq6.l5b2', 
           'kq6.r5b2',
           'kq7.l5b2',
           'kq7.r5b2',  
           'kq8.l5b2',
           'kq8.r5b2', 
           'kq9.l5b2', 
           'kq9.r5b2',
           'kq10.l5b2',
           'kq10.r5b2', 
           'kqtl11.r5b2', 
           'kqt12.r5b2', 
           'kqt13.r5b2',
           ]

def reset_delta_k(my_k_list):
        for kk in my_k_list:
                collider.vars[f'{kk}_delta'] = 0.000000

reset_delta_k(my_k_list)

for ii in my_k_list:
        collider.vars[f'{ii}_0'] = collider.vars[ii]._get_value()
        collider.vars[f'{ii}_delta'] = 0.000000
        if 'r1' in ii:
                collider.vars[ii] = collider.vars[f'{ii}_0'] + collider.vars[f'{ii}_delta']*collider.vars['i_wire_ip1.b2']/350
        if 'l1' in ii:
                collider.vars[ii] = collider.vars[f'{ii}_0'] + collider.vars[f'{ii}_delta']*collider.vars['i_wire_ip1.b2']/350
 
        if 'r5' in ii:
                collider.vars[ii] = collider.vars[f'{ii}_0'] + collider.vars[f'{ii}_delta']*collider.vars['i_wire_ip5.b2']/350
        if 'l5' in ii:
                collider.vars[ii] = collider.vars[f'{ii}_0'] + collider.vars[f'{ii}_delta']*collider.vars['i_wire_ip5.b2']/350


for ii in data['k_delta']:
    collider.vars[f'{ii}_delta'] = data['k_delta'][ii]

tw_ref = line.twiss(method='4d')


# %%
line.vars['i_wire_ip1.b2'] = data['i_wire_ip1.b2']
line.vars['i_wire_ip5.b2'] = data['i_wire_ip5.b2']
# %%

tw_new = line.twiss(method='4d')
plt.plot(tw_ref['s'], (tw_new['betx']- tw_ref['betx'])/tw_ref['betx'], label='ref')
plt.plot(tw_ref['s'], (tw_new['bety']- tw_ref['bety'])/tw_ref['bety'], label='ref')

# %%

print(tw_ref.qx, tw_new.qx)
print(tw_ref.qy, tw_new.qy)
# %%
