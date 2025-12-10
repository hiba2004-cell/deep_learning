import mysql from 'mysql2/promise';

export const pool = mysql.createPool({
    host: 'localhost',
    user: 'root',
    password: '',
    database: 'expression_besoin',
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
});

export async function query(text: string, params?: any[]) {
    const [rows] = await pool.query(text, params);
    return rows;
}