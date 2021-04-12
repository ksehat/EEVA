from pyflowchart import Flowchart as fc

with open('eeva_system2_v2.2.1.py') as f:
    code = f.read()

flow1 = fc.from_code(code)
print(flow1.flowchart())