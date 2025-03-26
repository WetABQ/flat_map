use flat_table::*;

fn main() {
    let mut table: Table<u64, u64> = Table::new_with_capacity(16);

    let n = 114514;

    for i in 0..n { 
        table.insert(i, i);
    }

    for i in 0..n {
        assert_eq!(table.get(&i), Some(&i), "failed at {}", i);
    }

}

