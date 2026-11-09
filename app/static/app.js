const input=document.getElementById("video"),run=document.getElementById("run"),msg=document.getElementById("message");
run.addEventListener("click",async()=>{if(!input.files.length){msg.textContent="Please select a video first.";return}
const data=new FormData();data.append("file",input.files[0]);run.disabled=true;msg.textContent="Analyzing video… first run may download OCR models.";
try{const res=await fetch("/api/analyze",{method:"POST",body:data});const body=await res.json();if(!res.ok)throw new Error(body.detail||"Analysis failed");
document.getElementById("frames").textContent=body.frames_processed;document.getElementById("count").textContent=body.plates.length;document.getElementById("duration").textContent=body.duration_seconds+"s";
const rows=document.getElementById("rows");rows.innerHTML="";
if(!body.plates.length)rows.innerHTML='<tr><td colspan="5" class="empty">No plate candidates recognized. Try clearer footage or a closer view.</td></tr>';
for(const p of body.plates){const tr=document.createElement("tr");[p.plate,(p.confidence*100).toFixed(1)+"%",p.first_seen_seconds+"s",p.last_seen_seconds+"s",p.appearances].forEach(v=>{const td=document.createElement("td");td.textContent=v;tr.appendChild(td)});rows.appendChild(tr)}
const out=document.getElementById("output");out.classList.remove("hidden");document.getElementById("resultVideo").src=body.annotated_video;msg.textContent="Analysis complete.";
}catch(e){msg.textContent=e.message}finally{run.disabled=false}});
