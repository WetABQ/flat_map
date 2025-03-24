use std::{hash::{BuildHasher, Hash, Hasher}, mem::MaybeUninit};

use foldhash::fast::RandomState;
use simd_itertools::PositionSimd;

const GROUP_SIZE: usize = 16;
const GROWTH_FACTOR_THRESHOLD: f64 = 0.85;

const EMPTY: u8 = 0xFF;
const DELETED: u8 = 0x80;

#[derive(Debug)]
pub struct Item<K: Eq + Hash, V> {
  hash: u64,
  key: K,
  value: V,
}

#[derive(Debug)]
pub struct Group<K: Eq + Hash, V> {
  items: [MaybeUninit<Item<K, V>>; GROUP_SIZE],
}

#[derive(Debug)]
pub struct Table<K: Eq + Hash, V, R: BuildHasher = RandomState> {
  groups: Vec<Group<K, V>>,
  control_bytes: Vec<u8>,
  hasher: R,
  capacity: usize,
  size: usize,
}

impl<K: Eq + Hash, V> Item<K, V> {
  fn new(hash: u64, key: K, value: V) -> Self {
    Self { hash, key, value }
  }
}

impl<K: Eq + Hash, V> Group<K, V> {
  fn new() -> Self {
    Self { items: [ const { MaybeUninit::uninit() }; GROUP_SIZE] }
  }
}

impl<K: Eq + Hash, V> Table<K, V, RandomState> {
  pub fn new_with_capacity(capacity: usize) -> Self {
    Self::new_with_capacity_and_hasher(capacity, RandomState::default())
  }

  pub fn new() -> Self {
    Self::new_with_capacity(32)
  }
  
}

impl<K: Eq + Hash, V, R: BuildHasher + Clone> Table<K, V, R> {

  pub fn new_with_capacity_and_hasher(capacity: usize, hasher: R) -> Self {
    let capacity = round_up_to_nearest_multiple_of_16(capacity);
    let group_capacity = capacity / 16;
    let groups = (0..group_capacity).map(|_| Group::new()).collect();
    Self { groups, control_bytes: vec![EMPTY; capacity],hasher: hasher, capacity: capacity, size: 0 }
  }

  pub fn insert(&mut self, key: K, value: V) {
    let hash = self.hash_u64(&key);
    let group_index = h1(hash) % self.groups.len();

    println!("<insert> hash: {}, h1: {}, h2: {}, group_index: {}, capacity: {}", hash, h1(hash), h2(hash), group_index, self.capacity);

    // Translate group index to control byte index
    let mut current_control_byte_idx = group_index * GROUP_SIZE;

    // Probe the control bytes by GROUP_SIZE
    loop {
      let view = &self.control_bytes[current_control_byte_idx..current_control_byte_idx + GROUP_SIZE];
      let idx_in_group = self.probe_empty_or_deleted(view);
      if let Some(idx_in_group) = idx_in_group {
        // Found an empty slot or deleted slot, insert the item
        let update_group_idx = current_control_byte_idx / GROUP_SIZE;
        println!("found empty slot at group_idx: {} idx: {}", update_group_idx, idx_in_group);
        self.groups[update_group_idx].items[idx_in_group].write(Item::new(hash, key, value));
        self.control_bytes[current_control_byte_idx + idx_in_group] = h2(hash);
        self.size += 1;
        if self.load_factor() > GROWTH_FACTOR_THRESHOLD {
          println!("growing table");
          self.grow();
        }
        break;
      }
      // Move to the next group
      current_control_byte_idx = (current_control_byte_idx + GROUP_SIZE) % self.control_bytes.len();
    }
  }

  pub fn get(&self, item: &K) -> Option<&V> {
    let hash = self.hash_u64(item);
    let group_idx = h1(hash) % self.groups.len();

    println!("<get> hash: {}, h1: {}, h2: {}, group_idx: {}", hash, h1(hash), h2(hash), group_idx);

    // Translate group index to control byte index
    let mut current_control_byte_idx = group_idx * GROUP_SIZE;
    
    // Probe the control bytes by GROUP_SIZE
    loop {
      let view = &self.control_bytes[current_control_byte_idx..current_control_byte_idx + GROUP_SIZE];
      let idx_in_group = self.probe_for_item(view, h2(hash));

      if let Some(idx_in_group) = idx_in_group {
        // Found the item
        let update_group_idx = current_control_byte_idx / GROUP_SIZE;
        println!("found item at group_idx: {} idx: {}", update_group_idx, idx_in_group);
        unsafe {
          return Some(&self.groups[update_group_idx].items[idx_in_group].assume_init_ref().value);
        }
      } else {
        // Item not found, check if the group is full
        println!("item not found in group_idx: {}", current_control_byte_idx / GROUP_SIZE);
        let empty_idx = self.probe_empty(view);
        if let Some(empty_idx) = empty_idx {
          // Found an empty slot, item is not in the table
          println!("found empty slot at group_idx: {} idx: {}, early exit", current_control_byte_idx / GROUP_SIZE, empty_idx);
          return None;
        }
        // No empty slot found, this group is full, continue to the next group
      }
      // Move to the next group
      current_control_byte_idx = (current_control_byte_idx + GROUP_SIZE) % self.control_bytes.len();
    }
  }

  fn probe_empty_or_deleted(&self, group_slice: &[u8]) -> Option<usize> {
    group_slice.iter().position_simd(|&control_byte| control_byte == EMPTY || control_byte == DELETED)
  }

  fn probe_for_item(&self, group_slice: &[u8], expected_control_byte: u8) -> Option<usize> {
    group_slice.iter().position_simd(|&control_byte| control_byte == expected_control_byte)
  }

  fn probe_empty(&self, group_slice: &[u8]) -> Option<usize> {
    group_slice.iter().position_simd(|&control_byte| control_byte == EMPTY)
  }

  fn grow(&mut self) {
    self.capacity *= 2;
    let new_groups: Vec<Group<K, V>> = (0..self.capacity / 16).map(|_| Group::new()).collect();
    self.groups.extend(new_groups);
    self.control_bytes.resize(self.capacity, 0);
  }

  fn hash_u64<T: Hash>(&self, item: &T) -> u64 {
    let mut hasher = self.hasher.build_hasher();

    item.hash(&mut hasher);
    hasher.finish()
  }

  pub fn load_factor(&self) -> f64 {
    self.size as f64 / self.capacity as f64
  }
}

impl<K: Eq + Hash, V, S: BuildHasher> IntoIterator for Table<K, V, S>
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V, S>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            table: self,
            current_group: 0,
            current_index: 0,
            items_yielded: 0,
        }
    }
}

impl<K: Eq + Hash, V, S: BuildHasher> Table<K, V, S>
{
    pub fn iter(&self) -> Iter<'_, K, V, S> {
        Iter {
            table: self,
            current_group: 0,
            current_index: 0,
            items_yielded: 0,
        }
    }
}

pub struct IntoIter<K: Eq + Hash, V, S: BuildHasher> {
    table: Table<K, V, S>,
    current_group: usize,
    current_index: usize,
    items_yielded: usize,
}

pub struct Iter<'a, K: Eq + Hash, V, S: BuildHasher> {
    table: &'a Table<K, V, S>,
    current_group: usize,
    current_index: usize,
    items_yielded: usize,
}

impl<'a, K: Eq + Hash, V, S: BuildHasher> Iterator for Iter<'a, K, V, S> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.items_yielded >= self.table.size {
            return None;
        }

        let mut current_control_byte_idx = self.current_group * GROUP_SIZE + self.current_index;

        while current_control_byte_idx < self.table.control_bytes.len() {
            
            while self.current_index < GROUP_SIZE {
                let control_byte = self.table.control_bytes[current_control_byte_idx];
                
                // Skip empty slots or deleted slots
                if control_byte != EMPTY && control_byte != DELETED {
                  let item = unsafe { self.table.groups[self.current_group].items[self.current_index].assume_init_ref() };
                  self.current_index += 1;
                  self.items_yielded += 1;
                  return Some((&item.key, &item.value));
                }

                self.current_index += 1;
                current_control_byte_idx += 1;
              }

            self.current_group += 1;
            self.current_index = 0;
        }
        
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.table.size - self.items_yielded;
        (remaining, Some(remaining))
    }
}

impl<K: Eq + Hash, V, S: BuildHasher> Iterator for IntoIter<K, V, S> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.items_yielded >= self.table.size {
            return None;
        }

        let mut current_control_byte_idx = self.current_group * GROUP_SIZE + self.current_index;

        while current_control_byte_idx < self.table.control_bytes.len() {
            
            while self.current_index < GROUP_SIZE {
                let control_byte = self.table.control_bytes[current_control_byte_idx];
                
                // Skip empty slots or deleted slots
                if control_byte != EMPTY && control_byte != DELETED {
                    self.items_yielded += 1;
                    // Check whether the item has been copied or not
                    let item = unsafe { self.table.groups[self.current_group].items[self.current_index].assume_init_read() };
                    return Some((item.key, item.value));
                }

                self.current_index += 1;
                current_control_byte_idx += 1;

            }

            self.current_group += 1;
            self.current_index = 0;
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.table.size - self.items_yielded;
        (remaining, Some(remaining))
    }
}

impl<K: Eq + Hash, V> FromIterator<(K, V)> for Table<K, V, RandomState>
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, upper) = iter.size_hint();
        
        // Use the upper bound if available, otherwise use the lower bound
        let capacity = upper.unwrap_or(lower);
        let mut table = Table::<K, V, RandomState>::new_with_capacity(capacity);
        
        for (key, value) in iter {
            table.insert(key, value);
        }
        
        table
    }
}


// Last 57 bits of the hash
fn h1(hash: u64) -> usize {
  (hash & 0x01FFFFFFFFFFFFFF) as usize
}

// Top 7 bits of the hash
fn h2(hash: u64) -> u8 {
  (hash >> 57) as u8
}

fn round_up_to_nearest_multiple_of_16(n: usize) -> usize {
  (n + 15) & !15
}