import json, re, sys
prompts=json.load(sys.stdin)
out=[]
for p in prompts:
 m=re.search(r"What is (\d+) (.) (\d+)", p); a,op,b=int(m.group(1)),m.group(2),int(m.group(3)); out.append(str({"+":a+b,"-":a-b,"*":a*b}[op]))
json.dump(out, sys.stdout)
