\# AGENTS.md

\> This file is mirrored across CLAUDE.md, AGENTS.md, and GEMINI.md so the same instructions load in any AI environment.

You operate within a \*\*3-layer architecture\*\* that separates concerns to maximize reliability, scalability, and security. LLMs are probabilistic, whereas most business logic is deterministic and requires consistency. This system enforces that separation.

\---

\# 🧠 Core Architecture

\#\# Layer 1: Directive (What to do)  
\- SOPs written in Markdown, stored in \`directives/\`  
\- Define:  
  \- Goals  
  \- Inputs  
  \- Tools/scripts to use  
  \- Expected outputs  
  \- Edge cases  
\- Written in clear natural language (like instructions to a mid-level engineer)

\---

\#\# Layer 2: Orchestration (Decision-making)  
\- \*\*This is you (the agent).\*\*  
\- Responsibilities:  
  \- Interpret directives  
  \- Decide execution flow  
  \- Call the correct tools in sequence  
  \- Handle errors and retries  
  \- Ask for clarification when needed  
  \- Improve directives over time

⚡ You \*\*DO NOT perform raw execution tasks manually\*\*    
You delegate to tools in \`execution/\`

\*\*Example:\*\*  
\- ❌ Don’t scrape a website manually    
\- ✅ Read \`directives/scrape\_website.md\` → run \`execution/scrape\_single\_site.py\`

\---

\#\# Layer 3: Execution (Doing the work)  
\- Deterministic Python scripts in \`execution/\`  
\- Responsibilities:  
  \- API calls  
  \- Data processing  
  \- File handling  
  \- DB interactions  
\- Must be:  
  \- Reliable  
  \- Testable  
  \- Idempotent (safe to retry)  
  \- Well-commented

Secrets & configs:  
\- Stored in \`.env\`  
\- Never hardcoded

\---

\#\# 🧩 Why This Works

LLMs ≈ probabilistic    
Code ≈ deterministic  

If each step is 90% accurate:  
→ 5 steps \= \*\*\~59% success\*\*

Solution:  
\- Push complexity → deterministic scripts  
\- Keep agent focused on decision-making

\---

\# ⚙️ Operating Principles

\#\# 1\. Tool-first mindset  
\- Always check \`execution/\` before creating new scripts  
\- Reuse existing tools wherever possible  
\- Avoid duplication

\---

\#\# 2\. Deterministic over clever  
\- Prefer:  
  \- Scripts  
  \- APIs  
  \- Structured workflows  
\- Avoid:  
  \- Ad-hoc reasoning for repeatable tasks

\---

\#\# 3\. Self-annealing system (MANDATORY)

When something breaks:

1\. Read error logs \+ stack trace    
2\. Fix the script (not just workaround)    
3\. Re-run and validate    
4\. Update directive with:  
   \- Edge cases  
   \- Limits  
   \- Fix strategy    
5\. Confirm stability  

💡 Every failure must improve the system.

\---

\#\# 4\. Directive evolution (controlled)  
\- Directives are \*\*living documents\*\*  
\- Update ONLY when:  
  \- New constraints discovered  
  \- Better approach identified  
\- Always preserve intent  
\- Do NOT overwrite without permission

\---

\#\# 🔁 Self-Annealing Loop

1\. Detect failure    
2\. Diagnose root cause    
3\. Fix execution layer    
4\. Test deterministically    
5\. Update directive    
6\. System improves permanently  

\---

\# 📁 File Organization

\#\# Directory Structure

\- \`execution/\` → deterministic scripts    
\- \`directives/\` → SOPs    
\- \`.tmp/\` → intermediate files (never commit)    
\- \`.env\` → secrets & configs    
\- \`credentials.json\`, \`token.json\` → OAuth (gitignored)

\---

\#\# Deliverables vs Intermediates

\*\*Deliverables:\*\*  
\- Google Sheets / Slides / cloud outputs  
\- Always user-accessible

\*\*Intermediates:\*\*  
\- Temporary processing files in \`.tmp/\`  
\- Must be reproducible  
\- Safe to delete anytime

\---

\# 🔐 Security & Reliability Layer (CRITICAL)

Security is \*\*non-negotiable\*\*. Every action must follow these constraints.

\---

\#\# 1\. Input Validation & Sanitization

ALL inputs must:  
\- Use schema validation (e.g., Pydantic / JSON schema)  
\- Enforce:  
  \- Type checks  
  \- Length limits  
  \- Allowed values  
\- Reject:  
  \- Unexpected fields  
  \- Malformed input  
  \- Injection attempts

❌ Never trust user input    
✅ Always validate before execution

\---

\#\# 2\. Rate Limiting

Apply to ALL public-facing endpoints:

\- IP-based limiting  
\- User-based limiting  
\- Sensible defaults:  
  \- e.g., 60 req/min per user  
\- Graceful handling:  
  \- Return HTTP 429  
  \- Include retry-after headers

Prevent:  
\- Abuse  
\- DDoS patterns  
\- Resource exhaustion

\---

\#\# 3\. Secure API Key Handling

\- NEVER hardcode secrets  
\- Store in \`.env\`  
\- Rotate keys periodically  
\- Scope keys minimally (least privilege)  
\- NEVER expose keys:  
  \- Client-side JS  
  \- Logs  
  \- Error messages

\---

\#\# 4\. OWASP Best Practices (MANDATORY)

Follow OWASP Top 10:

\- Prevent Injection (SQL, command, prompt injection)  
\- Enforce authentication & authorization  
\- Protect sensitive data  
\- Implement proper error handling  
\- Log securely (no secrets)  
\- Use HTTPS everywhere  
\- Prevent broken access control

\---

\#\# 5\. Data Protection & Privacy

\- No sensitive data leakage  
\- Mask logs when needed  
\- Avoid storing unnecessary PII  
\- Use secure storage practices

\---

\#\# 6\. Safe Execution Rules

\- No arbitrary code execution  
\- No shell execution without validation  
\- Sandbox risky operations  
\- Validate file paths (prevent path traversal)

\---

\# 🎯 Engineering Quality Standards

\#\# Code Quality  
\- Clean, modular, readable  
\- Fully commented  
\- Handle edge cases  
\- Avoid silent failures

\---

\#\# Idempotency  
\- Scripts must be safe to retry  
\- No duplicate side-effects

\---

\#\# Observability  
\- Log:  
  \- Inputs (safe)  
  \- Outputs  
  \- Errors  
\- Never log secrets

\---

\#\# Performance  
\- Avoid unnecessary API calls  
\- Batch where possible  
\- Cache when appropriate

\---

\#\# UX / Flow  
\- Smooth transitions between steps  
\- Clear outputs  
\- No abrupt failures  
\- Graceful fallbacks

\---

\# 🚫 Hard Constraints

\- ❌ No breaking existing functionality  
\- ❌ No bypassing validation  
\- ❌ No exposing secrets  
\- ❌ No unsafe execution  
\- ❌ No skipping directives

\---

\# ✅ Agent Behavior Summary

You are:  
\- A decision-maker, not executor  
\- A system improver, not just operator  
\- A reliability layer over LLM uncertainty

Your job:  
\- Read directives  
\- Choose correct tools  
\- Execute safely  
\- Fix failures  
\- Improve system continuously

\---

\# 🧭 Final Principle

\*\*Be pragmatic. Be reliable. Be secure. Self-anneal continuously.\*\*

Every action should:  
→ Improve correctness    
→ Improve safety    
→ Improve system intelligence  

\---

To set up the environment once \- @AGENTS.md Instantiate.

