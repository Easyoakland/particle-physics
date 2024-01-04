// TODO finer grain imports (ahash, parameterize by Entity, bevy_math)
use bevy::{
    ecs::entity::Entity,
    math::{Rect, Vec2},
    utils::{HashMap, HashSet},
};
use smallvec::SmallVec;

type Key = (i32, i32);
const SMALL_VEC_SIZE: usize = 5;

/// A spatial container that allows querying for entities that share one or more grid cell
#[derive(Debug, Clone)]
pub struct SparseGrid2d {
    map: HashMap<Key, SmallVec<[Entity; SMALL_VEC_SIZE]>>,
    tile_size: f32,
}

impl SparseGrid2d {
    pub fn new(tile_size: f32) -> Self {
        Self {
            map: Default::default(),
            tile_size,
        }
    }

    /// Insert an entity in the given aabb coordinates.
    ///
    /// This inserts it in all key space positions it is at least partially contained in.
    pub fn insert_aabb(&mut self, aabb: Rect, entity: Entity) {
        for key in KeyIter::new(self.tile_size, aabb) {
            self.map.entry(key).or_default().push(entity);
        }
    }

    /// Insert an entity at the given point coordinate
    pub fn insert_point(&mut self, point: Vec2, entity: Entity) {
        let key = self.key_from_point(point);
        self.map.entry(key).or_default().push(entity);
    }

    /// Get an iterator of the entities in the key space grid cells containing the given aabb.
    ///
    /// May contain duplicates if some entities are in more than one grid cell.
    pub fn aabb_iter(&self, aabb: Rect) -> impl Iterator<Item = Entity> + '_ {
        KeyIter::new(self.tile_size, aabb)
            .filter_map(|key| self.map.get(&key))
            .flatten()
            .copied()
    }

    /// Creates a hash set (deduplicates) with all the entities in the grid cells covered by the given aabb
    pub fn query_aabb(&self, aabb: Rect) -> HashSet<Entity> {
        self.aabb_iter(aabb).collect()
    }

    /// Get an iterator of all entities in the key space grid cell containing the given point
    pub fn point_iter(&'_ self, point: Vec2) -> impl Iterator<Item = Entity> + '_ {
        let key = self.key_from_point(point);

        self.map.get(&key).into_iter().flatten().copied()
    }

    /// Remove all entities from the map
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Remove all entities from the map, but keep the heap-allocated inner data structures
    pub fn soft_clear(&mut self) {
        for vec in self.map.values_mut() {
            vec.clear()
        }
    }

    /// Get the key tile containing the point.
    fn key_from_point(&self, point: Vec2) -> Key {
        // Because 0 is the first positive key space tile index and -1 is the first negative key space tile
        // `floor()` is used for the positive (in addition to the negative) instead of `ceil()`
        (
            (point.x / self.tile_size).floor() as i32,
            (point.y / self.tile_size).floor() as i32,
        )
    }
}

/// Iterator over [`Key`]s
struct KeyIter {
    width: u32,
    start: Key,
    /// 1d Idx of next key
    current: u32,
    /// Total number of key tiles to yield
    count: u32,
}

impl KeyIter {
    /// Make [`KeyIter`] over all keys that contain the aabb.
    /// If a key space location partially contains the aabb it is included in the iterator.
    fn new(tile_size: f32, aabb: Rect) -> Self {
        let Rect { min, max } = aabb.into();
        // convert to key space
        let s = tile_size;
        // by flooring the min and ceiling the max here it should include the key slot partially covered by the Rect.
        // the opposite (floor max, ceil min) would exclude the key slot partially covered by the Rect.
        let min = ((min.x / s).floor() as i32, (min.y / s).floor() as i32);
        let max = ((max.x / s).ceil() as i32, (max.y / s).ceil() as i32);
        let width = max.0.abs_diff(min.0);
        let height = max.1.abs_diff(min.1);
        let count = width * height;
        Self {
            start: min,
            current: 0,
            width,
            count,
        }
    }
}

impl Iterator for KeyIter {
    type Item = Key;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current;
        self.current += 1;

        if current < self.count {
            Some((
                self.start
                    .0
                    .checked_add_unsigned(current.rem_euclid(self.width))
                    .expect("signed + unsigned overflow"),
                self.start
                    .1
                    .checked_add_unsigned(current / self.width)
                    .expect("signed + unsigned overflow"),
            ))
        } else {
            None
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let x = (self.count - self.current).try_into().unwrap();
        (x, Some(x))
    }
}
impl ExactSizeIterator for KeyIter {}

#[cfg(test)]
mod tests {
    use bevy::math::vec2;
    use bevy::utils::HashSet;

    use super::*;

    const TILE_SIZE: f32 = 1.0;

    #[test]
    fn keys_single() {
        let keys: Vec<Key> = KeyIter::new(
            TILE_SIZE,
            Rect {
                min: vec2(0.001, 0.001),
                max: vec2(0.001, 0.001),
            },
        )
        .collect();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0], (0, 0));
    }

    #[test]
    fn keys_four_around_origin() {
        let keys: Vec<Key> = KeyIter::new(
            TILE_SIZE,
            Rect {
                min: vec2(-0.001, -0.001),
                max: vec2(0.001, 0.001),
            },
        )
        .collect();
        assert!(keys.contains(&(0, 0)));
        assert!(keys.contains(&(0, -1)));
        assert!(keys.contains(&(-1, 0)));
        assert!(keys.contains(&(-1, -1)));
        assert_eq!(keys.len(), 4);
    }

    #[test]
    fn matches() {
        let entity = Entity::from_raw(123);
        let mut db = SparseGrid2d::new(TILE_SIZE);
        db.insert_aabb(
            Rect {
                min: vec2(-0.001, -0.001),
                max: vec2(0.001, 0.001),
            },
            entity,
        );

        let matches: Vec<Entity> = db
            .aabb_iter(Rect {
                min: vec2(0.001, 0.001),
                max: vec2(0.001, 0.001),
            })
            .collect();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0], entity);
    }

    #[test]
    fn key_negative() {
        let h = TILE_SIZE / 2.0;
        let keys: Vec<Key> = KeyIter::new(
            TILE_SIZE,
            Rect {
                min: vec2(-h, -h),
                max: vec2(-h, -h),
            },
        )
        .collect();
        assert!(keys.contains(&(-1, -1)));
        assert_eq!(keys.len(), 1);
    }
    #[test]
    fn query_points() {
        let mut db = SparseGrid2d::new(TILE_SIZE);
        let e1 = Entity::from_raw(1);
        let e2 = Entity::from_raw(2);
        db.insert_point(vec2(0.5, 0.5), e1);
        db.insert_point(vec2(0.499, 0.501), e2);

        let matches: HashSet<_> = db.point_iter(vec2(0.499, 0.501)).collect();
        assert!(matches.contains(&e1));
        assert!(matches.contains(&e2));
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn matches_complex() {
        let h = TILE_SIZE / 2.0;
        let e1 = Entity::from_raw(1);
        let e2 = Entity::from_raw(2);
        let e3 = Entity::from_raw(3);
        let mut db = SparseGrid2d::new(TILE_SIZE);
        db.insert_aabb(
            Rect {
                min: vec2(-h, -h),
                max: vec2(h, h),
            },
            e1,
        );
        db.insert_aabb(
            Rect {
                min: vec2(h, h),
                max: vec2(h, h),
            },
            e2,
        );
        db.insert_aabb(
            Rect {
                min: vec2(-h, -h),
                max: vec2(-h, -h),
            },
            e3,
        );

        let matches: Vec<Entity> = db
            .aabb_iter(Rect {
                min: vec2(-h, -h),
                max: vec2(h, h),
            })
            .collect();
        // assert_eq!(matches.len(), 3);
        assert!(matches.contains(&e1));
        assert!(matches.contains(&e2));
        assert!(matches.contains(&e3));

        let matches = db.query_aabb(Rect {
            min: vec2(-0.001, -0.001),
            max: vec2(-0.001, -0.001),
        });
        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&e1));
        assert!(matches.contains(&e3));

        let matches: Vec<Entity> = db
            .aabb_iter(Rect {
                min: vec2(-0.001, -0.001),
                max: vec2(-0.001, -0.001),
            })
            .collect();
        assert_eq!(matches[0], e1);
    }

    #[test]
    fn query_points_tilesize_10() {
        let mut db = SparseGrid2d::new(10.);
        let e1 = Entity::from_raw(1);
        let e2 = Entity::from_raw(2);
        let e3 = Entity::from_raw(3);
        db.insert_point(vec2(12f32, 15f32), e1);
        db.insert_point(vec2(15f32, 12f32), e2);
        db.insert_point(vec2(15f32, 20f32), e3);
        let matches: HashSet<_> = db.point_iter(vec2(19.9, 19.9)).collect();
        assert!(matches.contains(&e1));
        assert!(matches.contains(&e2));
        assert!(!matches.contains(&e3));
        assert_eq!(matches.len(), 2);
    }
}
