use flat_table::*;

fn main() {
    let mut table: Table<u64, u64> = Table::new_with_capacity(32);
    table.insert(1, 1);
    table.insert(2, 2);
    table.insert(3, 3);
    table.insert(4, 4);
    table.insert(5, 5);
    table.insert(6, 6);
    
    for item in table.iter() {
        println!("{:?}", item);
    }

}

