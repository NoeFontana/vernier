//! Paradigm-agnostic merge policy: image-id partition disjointness
//! and strict-mode rank-collision detection.
//!
//! Each paradigm's outer merge accumulator embeds a
//! [`BaseMergeAccumulator`] and delegates the policy checks to its
//! [`BaseMergeAccumulator::ingest_image_ids`] and
//! [`BaseMergeAccumulator::ingest_rank_id`] methods, then folds in its
//! own paradigm-specific cell store.

use std::collections::{HashMap, HashSet};

/// `RankId` type alias re-exported here so paradigm crates don't have
/// to import from `envelope` separately.
pub type RankId = u32;

/// Sentinel `RankId` carried in the partition-overlap error when one
/// of the colliding partials lacked a rank id (single-rank flow). Real
/// `rank_id`s should always be `< u32::MAX` in any DDP shape.
pub(crate) const UNRANKED_SENTINEL: RankId = u32::MAX;

use crate::error::PartialError;

/// Cross-partial state that every paradigm's merge accumulator
/// shares: image-id ownership (for the disjoint-partition rule) and
/// strict-mode rank-id distinctness.
///
/// Paradigms wrap this in their own outer accumulator and call
/// [`Self::ingest_rank_id`] + [`Self::ingest_image_ids`] from their
/// `ingest` per partial, before folding the paradigm-specific body.
#[derive(Debug)]
pub struct BaseMergeAccumulator {
    /// Image id → rank that ingested it. Source of truth for the
    /// merged `seen_images` set.
    pub image_owner: HashMap<i64, RankId>,
    /// Rank ids observed so far. Only populated in strict mode (the
    /// distinctness invariant is strict-only); empty in corrected.
    pub seen_rank_ids: HashSet<RankId>,
    /// Whether the receiver is in strict mode.
    pub strict: bool,
}

impl BaseMergeAccumulator {
    /// Construct an empty accumulator. `strict` controls whether the
    /// rank-id distinctness check fires.
    pub fn new(strict: bool) -> Self {
        Self {
            image_owner: HashMap::new(),
            seen_rank_ids: HashSet::new(),
            strict,
        }
    }

    /// Record one partial's `rank_id`. In strict mode, returns
    /// [`PartialError::RankCollision`] if a previous partial declared
    /// the same id. In corrected mode this is a no-op.
    pub fn ingest_rank_id(&mut self, rank_id: Option<RankId>) -> Result<(), PartialError> {
        if !self.strict {
            return Ok(());
        }
        if let Some(rid) = rank_id {
            if !self.seen_rank_ids.insert(rid) {
                return Err(PartialError::RankCollision { rank_id: rid });
            }
        }
        Ok(())
    }

    /// Record one partial's `seen_images` against its declared
    /// `rank_id`. Returns [`PartialError::PartitionOverlap`] if any
    /// `image_id` was already registered by a different rank.
    pub fn ingest_image_ids(
        &mut self,
        rank_id: Option<RankId>,
        image_ids: impl IntoIterator<Item = i64>,
    ) -> Result<(), PartialError> {
        let owner = rank_id.unwrap_or(UNRANKED_SENTINEL);
        for id in image_ids {
            if let Some(&prev) = self.image_owner.get(&id) {
                let (a, b) = (prev.min(owner), prev.max(owner));
                return Err(PartialError::PartitionOverlap {
                    rank_a: a,
                    rank_b: b,
                    image_id: id,
                });
            }
            self.image_owner.insert(id, owner);
        }
        Ok(())
    }

    /// Borrow the merged image-id set. Order is unspecified — sort
    /// before consuming if determinism is needed.
    pub fn image_ids(&self) -> impl Iterator<Item = i64> + '_ {
        self.image_owner.keys().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_collision_strict() {
        let mut acc = BaseMergeAccumulator::new(true);
        acc.ingest_rank_id(Some(0)).unwrap();
        let err = acc.ingest_rank_id(Some(0)).unwrap_err();
        assert!(matches!(err, PartialError::RankCollision { rank_id: 0 }));
    }

    #[test]
    fn rank_collision_corrected_tolerated() {
        let mut acc = BaseMergeAccumulator::new(false);
        acc.ingest_rank_id(Some(0)).unwrap();
        // Corrected mode tolerates duplicate rank_ids — they're informational only.
        acc.ingest_rank_id(Some(0)).unwrap();
    }

    #[test]
    fn partition_overlap_named_ranks() {
        let mut acc = BaseMergeAccumulator::new(true);
        acc.ingest_image_ids(Some(0), [1, 2, 3]).unwrap();
        let err = acc.ingest_image_ids(Some(1), [3, 4]).unwrap_err();
        match err {
            PartialError::PartitionOverlap {
                rank_a,
                rank_b,
                image_id,
            } => {
                assert_eq!(rank_a, 0);
                assert_eq!(rank_b, 1);
                assert_eq!(image_id, 3);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn partition_overlap_unranked_sentinel() {
        let mut acc = BaseMergeAccumulator::new(false);
        acc.ingest_image_ids(None, [7]).unwrap();
        let err = acc.ingest_image_ids(None, [7]).unwrap_err();
        match err {
            PartialError::PartitionOverlap { rank_a, rank_b, .. } => {
                assert_eq!(rank_a, UNRANKED_SENTINEL);
                assert_eq!(rank_b, UNRANKED_SENTINEL);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
