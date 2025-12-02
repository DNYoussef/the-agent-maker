# V1 Documentation Archive Plan

**Purpose**: Identify and archive inaccurate/outdated V1 documentation to prevent confusion

**Date**: 2025-10-16

**Status**: Ready for execution

---

## Overview

After completing the V2 documentation reconciliation, several V1 documents contain **inaccurate or outdated information** that could mislead developers. These need to be archived with clear warnings.

**Strategy**: Move inaccurate documents to `v1-reference/archive/` with `V1_OLD_` prefix and add deprecation warnings.

---

## Files to Archive

### ❌ CRITICAL: Phase 7 V1 Guide (Archived)

**File**: `phases/phase7/PHASE7_COMPLETE_GUIDE_V1_OLD.md`

**Status**: ✅ Already archived (has V1_OLD suffix)

**Reason**: Describes manual ADAS with N=4 fixed experts. V2 is self-guided with N=3-10 model-determined experts.

**Action**: Already correctly named, no action needed.

---

### ⚠️ NO FILES NEED ARCHIVING

**Discovery**: After comprehensive analysis, **NO additional files need archiving** because:

1. **V1 Implementation Docs** (`v1-reference/implementation/`) - Already in V1 reference directory
2. **Phase 1-4 Guides** - Accurate for both V1 and V2 (no redesign)
3. **Phase 5-8 Guides** - Already document V2 systems correctly
4. **CLAUDE.md** - ✅ Just updated with correct V2 descriptions

---

## Verification: Why No Additional Archives Needed

### Phase 5 Documentation
- `phases/phase5/LOGICAL_UNDERSTANDING.md` → ✅ Links to V2 file correctly
- `phases/phase5/PHASE5_LOGICAL_UNDERSTANDING_V2.md` → ✅ V2 complete spec
- `phases/phase5/PHASE5_EUDAIMONIA_SYSTEM.md` → ✅ V2 complete (589 lines)
- **No V1-specific Phase 5 docs exist** → Nothing to archive

### Phase 6 Documentation
- `phases/phase6/LOGICAL_UNDERSTANDING.md` → ✅ Describes V2 A/B cycles correctly
- `phases/phase6/PHASE6_COMPLETE_GUIDE.md` → ✅ V2 implementation (70.1% complete)
- **No V1-specific Phase 6 docs exist** → Nothing to archive

### Phase 7 Documentation
- `phases/phase7/LOGICAL_UNDERSTANDING.md` → ✅ Describes V2 self-guided system
- `phases/phase7/PHASE7_COMPLETE_GUIDE.md` → ✅ V2 implementation
- `phases/phase7/PHASE7_COMPLETE_GUIDE_V1_OLD.md` → ✅ Already archived (V1_OLD suffix)
- **Only V1 doc already has archive naming** → No action needed

### Phase 8 Documentation
- `phases/phase8/PHASE8_COMPLETE_GUIDE.md` → ✅ Production-ready V2 (1080 lines)
- `phases/phase8/PHASE8_BENCHMARK_TESTING.md` → ✅ V2 quality validation
- **No V1-specific Phase 8 docs exist** → Nothing to archive

---

## What We DID Update (Not Archive)

### ✅ Updated (Not Archived)

1. **CLAUDE.md** Lines 129-175
   - **Action**: Updated in-place to correct V2 descriptions
   - **Why not archived**: Master project documentation, needs to be current
   - **Status**: ✅ Complete

---

## Documentation Added (Not Archived)

### ✅ New V2 Documentation Created

1. **docs/v2-planning/PHASE5-8_V1_VS_V2_RECONCILIATION.md**
   - Explains V1 vs V2 differences
   - Prevents confusion about conflicting documentation

2. **docs/v2-planning/PHASE4_TO_PHASE5_HANDOFF.md**
   - Solves BitNet → Curriculum compatibility

3. **docs/DEPENDENCY_VERSIONS.md**
   - Pins all dependencies with frontier models

4. **docs/v2-planning/PHASE5_DREAM_CONSOLIDATION_IMPLEMENTATION.md**
   - Dream consolidation implementation based on paper

---

## Archive Decision Matrix

| File | Accurate? | Action | Reason |
|------|-----------|--------|--------|
| **CLAUDE.md** | ❌ Was inaccurate (Phase 5-8) | ✅ Updated in-place | Master doc, needs to be current |
| **Phase 1-4 guides** | ✅ Accurate | None | No redesign V1→V2 |
| **Phase 5 LOGICAL_UNDERSTANDING_V2.md** | ✅ Accurate V2 | None | Already V2-specific |
| **Phase 6 LOGICAL_UNDERSTANDING.md** | ✅ Accurate V2 | None | Describes V2 correctly |
| **Phase 7 PHASE7_COMPLETE_GUIDE_V1_OLD.md** | ❌ Outdated V1 | ✅ Already archived | Has V1_OLD suffix |
| **Phase 8 PHASE8_COMPLETE_GUIDE.md** | ✅ Accurate V2 | None | Production-ready V2 |
| **v1-reference/implementation/** | ✅ Accurate V1 | None | Already in V1 directory |

---

## Conclusion

**Archive Status**: ✅ **COMPLETE**

**Reason**: The only inaccurate V1 file (`PHASE7_COMPLETE_GUIDE_V1_OLD.md`) is **already archived** with proper V1_OLD naming.

**No additional archiving needed** because:
1. V1 implementation docs are already in `v1-reference/` directory
2. Phase-specific docs already describe V2 correctly
3. CLAUDE.md was updated in-place (correct approach for master docs)
4. New reconciliation doc (`PHASE5-8_V1_VS_V2_RECONCILIATION.md`) explains differences

---

## Prevention Strategy

### How to Avoid Future Confusion

1. **Clear V2 Labels**: All Phase 5-8 docs now have "V2 REDESIGN" labels
2. **Reconciliation Doc**: `PHASE5-8_V1_VS_V2_RECONCILIATION.md` explains differences
3. **Handoff Docs**: `PHASE4_TO_PHASE5_HANDOFF.md` solves compatibility
4. **Master Doc Updated**: CLAUDE.md now accurate for V2

### When to Archive in Future

**Archive if**:
- ✅ File describes V1 system and V2 has completely different approach
- ✅ File is in phase directory (not `v1-reference/`)
- ✅ File could mislead developers implementing V2

**Don't archive if**:
- ❌ File is already in `v1-reference/` directory (already clearly V1)
- ❌ File is master documentation (update in-place instead)
- ❌ File accurately describes V2 (no need to archive)

---

## Verification Commands

```bash
# Find any remaining V1-specific files (should find none except v1-reference/)
grep -r "V1 (Old)" phases/ docs/ --exclude-dir=v1-reference

# Find files with "9 personas" (V1 Phase 6 reference - should find only in reconciliation doc)
grep -r "9 specialized agents" phases/phase6/ docs/

# Find files with "BitNet + Grokfast" (V1 Phase 5 reference - should find only in reconciliation doc)
grep -r "Combined BitNet + Grokfast training" phases/phase5/ docs/

# Verify PHASE7_COMPLETE_GUIDE_V1_OLD.md is properly named
ls phases/phase7/*V1_OLD*
```

**Expected Results**: All searches should only find references in:
1. `v1-reference/` directory (intentional V1 docs)
2. `PHASE5-8_V1_VS_V2_RECONCILIATION.md` (explains differences)
3. `PHASE7_COMPLETE_GUIDE_V1_OLD.md` (already archived)

---

## Summary

✅ **All inaccurate documentation has been addressed**:
- CLAUDE.md updated to V2
- Phase 7 V1 guide already archived
- No other V1-specific files in phase directories

✅ **Prevention measures in place**:
- Reconciliation doc explains V1 vs V2
- Master docs updated with accurate V2 info
- Clear labeling ("V2 REDESIGN") on all Phase 5-8 docs

✅ **Repository is now V2-accurate** and ready for implementation.

---

**Related Documents**:
- [docs/v2-planning/PHASE5-8_V1_VS_V2_RECONCILIATION.md](PHASE5-8_V1_VS_V2_RECONCILIATION.md) - V1 vs V2 comparison
- [CLAUDE.md](../../CLAUDE.md) - Updated master documentation
- [phases/phase7/PHASE7_COMPLETE_GUIDE_V1_OLD.md](../../phases/phase7/PHASE7_COMPLETE_GUIDE_V1_OLD.md) - Archived V1 guide
