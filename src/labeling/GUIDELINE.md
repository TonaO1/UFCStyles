# Labeling Guideline: Martial-Arts Background

**Write this BEFORE you label anything.** One page per probe, with decision rules and examples.

## Categories

### 1. Wrestler

**Decision rule:** Primary offensive and defensive method is wrestling/grappling. Credentials: college/Olympic wrestler, heavy reliance on takedowns, control-based gameplan.

**Examples:**

- Colby Covington: D1 wrestler, bases entire game on takedown + top control
- Tyron Woodley: D1 wrestler, uses wrestling to control dominant strikers
- Ben Askren: Took down elite strikers repeatedly via wrestling

**Non-examples:**

- Jon Jones (strong wrestling but primarily striker / grappler hybrid)
- Khabib (wrestling base but overwhelmingly grapples on ground)

---

### 2. Striker: Boxing

**Decision rule:** Standalone boxing is rare in modern UFC. Boxing-focused credentials (amateur boxing record, professional box career) with limited kicks or grappling.

**Examples:**

- Anderson Silva: Olympic boxing, bases combinations around punches
- Anthony Smith: Heavy professional boxing background

**Non-examples:**

- Max Holloway (striker, but includes kicks and grappling)
- Tyron Woodley (has boxing but wrestler-primary)

---

### 3. Striker: Kickboxing / Muay Thai

**Decision rule:** Uses kicks, clinch work, or Thai clinch. Often European/Asian background. Muay Thai or kickboxing credentials.

**Examples:**

- Joanna Jędrzejczyk: Professional MMA striker with Dutch kickboxing background
- Mirko Cro Cop: Kickboxing base, elite kick game
- Overeem: Heavyweight striker/clincher with Dutch MT background

**Non-examples:**

- Anthony Pettis (striker but more boxing-based)
- Mike Perry (athlete who learned striking, not credentials)

---

### 4. BJJ / Grappler

**Decision rule:** Primary game is submission or guard-based jiu-jitsu. Often has high-level BJJ (purple belt minimum), competes in submission-heavy events. Ground game is primary plan, not secondary.

**Examples:**

- Charles Oliveira: Submission artist, elite guard, built entire game on BJJ
- Demian Maia: Grappler-based, goes to guard, uses position to grind
- Nate Diaz: Guard-focused, active on bottom

**Non-examples:**

- Khabib (grapples, but not BJJ-based; submission rate low)
- Robert Whittaker (can grapple, but not the primary)

---

### 5. Hybrid

**Decision rule:** No single martial art is dominant. Two or more systems are approximately equal in execution. Often well-rounded, uses different approaches per opponent.

**Examples:**

- Israel Adesanya: Striker + grappling, opponent-adaptive
- Dominick Reyes: Striker + wrestler, can deploy both
- Max Holloway: Boxer + striker + defensive wrestler, fully rounded

**Non-examples:**

- Khabib (70% wrestling, 30% other = not hybrid)
- Anderson Silva (70% boxing, 30% other = not hybrid)

---

### 6. Unclear

**Decision rule:** Insufficient information, mixed credentials, or the fighter has changed systems significantly. Or you genuinely cannot decide.

**Examples:**

- A fighter with 3 UFC fights and no clear pattern
- A young fighter with no documented background
- A fighter who switched entirely between eras

---

## Labeling Instructions

1. **Read the fighter's Wikipedia and Sherdog page.** Look for:
   - Amateur credentials (wrestling record, boxing record, BJJ belt)
   - Professional record in other sports (Muay Thai, boxing, grappling)
   - Coaching staff and camp style
   - Methods of victories (submission %, takedown rate %, decision rate %)

2. **Watch 30–60 seconds of two representative highlights.** You are checking:
   - What do they _choose_ to do when given a choice?
   - What are they _dangerous_ at?
   - What do they seem to _prefer_ and _train_?

3. **Assign one category.** If you are unsure between two, pick the one you are more confident in. If you are 50/50, use **Unclear**.

4. **Document your reasoning.** On the spreadsheet, note which source (Wikipedia, Sherdog, video) drove your decision.

---

## Reliability Check

After labeling ~50 fighters, re-label 20 at random **one week later** without looking at your previous labels.

Compute Cohen's kappa:

```
agreements = sum(label_time1 == label_time2)
kappa = (agreements / n) - expected_by_chance
```

Target: κ > 0.60. If κ < 0.50, revise the guideline and re-label all 20.

---

## Notes

- **Bias awareness:** Do not let recent results bias your category choice. A dominant wrestler who just got knocked out is still a wrestler.
- **Context matters:** The same fighter might be striker-based at lightweight and grappler-based at middleweight. Label based on their _primary_ division and gameplan.
- **Evolving fighters:** If a fighter's record shows clear shift (3 years wrestling → switch to kickboxing), use **Hybrid** or the current primary style, not the old one.
