package com.cs478.project;

import java.lang.reflect.Field;
import java.time.LocalTime;
import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class BaseRecord implements CSV<BaseRecord> {

    public long customer_ID;
    public long shopping_pt;
    public long record_type;
    public long day;
    public LocalTime time;
    public String state;
    public long location;
    public long group_size;
    public long homeowner;
    public long car_age;
    public String car_value;
    public long risk_factor;
    public long age_oldest;
    public long age_youngest;
    public long married_couple;
    public long C_previous;
    public long duration_previous;
    public long A;
    public long B;
    public long C;
    public long D;
    public long E;
    public long F;
    public long G;
    public long cost;


    public BaseRecord(String[] values) {
        this(
                toLong(values[0]),
                toLong(values[1]),
                toLong(values[2]),
                toLong(values[3]),
                values[4],
                values[5],
                toLong(values[6]),
                toLong(values[7]),
                toLong(values[8]),
                toLong(values[9]),
                values[10],
                toLong(values[11]),
                toLong(values[12]),
                toLong(values[13]),
                toLong(values[14]),
                toLong(values[15]),
                toLong(values[16]),
                toLong(values[17]),
                toLong(values[18]),
                toLong(values[19]),
                toLong(values[20]),
                toLong(values[21]),
                toLong(values[22]),
                toLong(values[23]),
                toLong(values[24])
        );
    }

    public BaseRecord(long customer_id, long shopping_pt, long record_type, long day, String time, String state, long location, long group_size, long homeowner, long car_age, String car_value, long risk_factor, long age_oldest, long age_youngest, long married_couple, long c_previous, long duration_previous, long a, long b, long c, long d, long e, long f, long g, long cost) {
        this.customer_ID = customer_id;
        this.shopping_pt = shopping_pt;
        this.record_type = record_type;
        this.day = day;
        this.location = location;
        this.group_size = group_size;
        this.homeowner = homeowner;
        this.car_age = car_age;
        this.risk_factor = risk_factor;
        this.age_oldest = age_oldest;
        this.age_youngest = age_youngest;
        this.married_couple = married_couple;
        this.C_previous = c_previous;
        this.duration_previous = duration_previous;
        this.A = a;
        this.B = b;
        this.C = c;
        this.D = d;
        this.E = e;
        this.F = f;
        this.G = g;
        this.cost = cost;
        this.time = LocalTime.parse(time);
        this.state = state;
        this.car_value = car_value;
    }

    public Stream<String> fields() {
        return Arrays.stream(BaseRecord.class.getFields())
                .map(Field::getName);
    }

    @Override
    public Stream<String> values() {
        return Arrays.stream(BaseRecord.class.getFields()).map(this::fieldToString);
    }

    @Override
    public String header() {
        return fields().collect(Collectors.joining(","));
    }

    @Override
    public String toString() {
        return values().collect(Collectors.joining(","));
    }

    private String fieldToString(Field field){
        try {
            if (field.getType().equals(long.class)) {
                long value = field.getLong(this);
                return value == -1 ? "NA" : Long.toString(value);
            }

            return field.get(this).toString();
        } catch (IllegalAccessException e) {
            return "NA";
        }
    }

    private static long toLong(String value) {
        if (value == null || "NA".equals(value))
            return -1;
        return Long.parseLong(value);
    }

    @Override
    public int compareTo(BaseRecord o) {
        int c = (int) (customer_ID - o.customer_ID);
        if(c != 0)
            return c;

        return (int) (shopping_pt - o.shopping_pt);
    }
}
