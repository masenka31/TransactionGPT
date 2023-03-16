using DrWatson
using CSV, DataFrames

# TODO: add age: should be probably age at day 1.1.1999 or something like that
# TODO: figure out if the orders / loans are necessary to include
# TODO: datetime columns and timestamp for the transaction date

function extract_info(num::Int)
    year = 1900 + div(num, 10000)
    month_num = mod(div(num, 100), 100)
    day = mod(num, 100)

    if month_num < 13
        month = month_num
        gender = "male"
    else
        month = month_num - 50
        gender = "female"
    end
    
    return (year, month, day, gender)
end

format_date(year::Int, month::Int, day::Int) = string(year, "-", lpad(string(month), 2, "0"), "-", lpad(string(day), 2, "0"))

function format_date(num::Int)
    year = 1900 + div(num, 10000)
    month = mod(div(num, 100), 100)
    day = mod(num, 100)

    return format_date(year, month, day)
end

function load_czech_banking_dataset()
    if !isfile(datadir("czech_banking_dataset.csv"))
        @info "The Czech banking dataset file does not yet exist. Doing the export."
        join_czech_dataset_tables()
    end
    
    return CSV.read(datadir("czech_banking_dataset.csv"), DataFrame)
end

function join_czech_dataset_tables()
    if !isdir(datadir("czech_banking_dataset"))
        error("""
            The dataset is not unziped - please, extract the individual .csv files
            from czech_banking_dataset.zip to a `czech_banking_dataset` folder."""
        )
    end
    
    account = CSV.read(datadir("czech_banking_dataset/account.csv"), DataFrame)
    card = CSV.read(datadir("czech_banking_dataset/card.csv"), DataFrame)
    client = CSV.read(datadir("czech_banking_dataset/client.csv"), DataFrame)
    disp = CSV.read(datadir("czech_banking_dataset/disp.csv"), DataFrame)
    district = CSV.read(datadir("czech_banking_dataset/district.csv"), DataFrame)
    loan = CSV.read(datadir("czech_banking_dataset/loan.csv"), DataFrame)
    order = CSV.read(datadir("czech_banking_dataset/order.csv"), DataFrame)
    trans = CSV.read(datadir("czech_banking_dataset/trans.csv"), DataFrame)
    
    @info "Dataframes loaded, joining the tables."
    
    # account processing
    account.date = map(x -> format_date(x), account.date)
    rename!(account, :date => :account_creation_date)

    # card processing
    card.issued = map(x -> format_date(parse(Int, x[1:6])), card.issued)
    rename!(card, :type => :card_type, :issued => :issued_date)

    # client processing
    # add new columns for birthday and gender to the client dataframe
    client.birthday = map(x -> format_date(extract_info(x)[1:3]...), client.birth_number)
    client.gender = map(x -> extract_info(x)[4], client.birth_number)

    # disp, district, loan processing
    rename!(disp, :type => :account_type)
    rename!(
        district,
        [:id, :district_name, :region, :no_inhabitans, :no_inhabitants499, :no_inhabitants1999,
        :no_inhabitants9999, :no_inhabitants10000, :no_cities, :urban_ratio, :avg_salary,
        :unemployment_rate1995, :unemployment_rate1999, :no_enterpreneurs, :crimes1995, :crimes1996]
    )
    loan.loan_date = map(x -> format_date(x), loan.date)
    rename!(loan, :amount => :loan_amount, :duration => :loan_duration, :payments => :monthly_payment, :status => :loan_status)

    # 'A' stands for contract finished, no problems
    # 'B' stands for contract finished, loan not payed
    # 'C' stands for running contract, OK thus-far
    # 'D' stands for running contract, client in debt

    # transactions processing and renaming
    trans.date = map(x -> format_date(x), trans.date)
    rename!(trans, :account => :counterparty_id, :type => :direction, :k_symbol => :customer_reference, :operation => :type)
    trans.direction = convert.(String, trans.direction)

    trans[trans[:, :direction] .== "PRIJEM", :direction] .= "inbound"
    trans[trans[:, :direction] .== "VYDAJ", :direction] .= "outbound"
    trans[trans[:, :direction] .== "VYBER", :direction] .= "outbound"

    for (orig, new) in [
            ["VYBER", "ATM withdrawal"],
            ["VKLAD", "Cash in"],
            ["PREVOD NA UCET", "Wire"],
            ["PREVOD Z UCTU", "Wire"],
            ["VYBER KARTOU", "Card"],
            ["", "Interest"]
        ]
        trans[trans[:, :type] .== orig, :type] .= new
    end

    ### JOINS ###
    ### first, join information about clients and accounts
    client_disp = innerjoin(disp, client, on=:client_id)
    client_account = leftjoin(client_disp, account, on=:account_id, makeunique=true)
    select!(client_account, Not([:district_id, :district_id_1, :frequency, :birth_number]))

    # make the shorter version of clien accounts -> two owners of one account to one df line
    owner = filter(:account_type => x -> x == "OWNER", client_account)
    select!(owner, Not(:account_type))
    disponent = filter(:account_type => x -> x == "DISPONENT", client_account)
    rename!(disponent, :client_id => :disponent_client_id, :birthday => :disponent_birthday, :gender => :disponent_gender)
    select!(disponent, Not([:account_creation_date, :account_type]))

    customer = outerjoin(owner, disponent, on=:account_id, makeunique=true)
    customer = leftjoin(customer, card, on=:disp_id)

    full_table = leftjoin(trans, customer, on=:account_id)
    select!(full_table, Not([:disp_id, :disp_id_1]))

    @info "Tables processed and joined, saving the result in $(datadir("czech_banking_dataset.csv"))."
    CSV.write(datadir("czech_banking_dataset.csv"), full_table)
end

"""
# for each account, we have people who can access the account
# there are accounts with multiple holders, these are probably married couples
g = groupby(client_account, :account_id)
fdf = filter(x -> nrow(x) > 1, g)
fdf = vcat(fdf...)

# this gives interesting results...
# there are not that many more males as owners than females...
countmap([[x, y] for (x, y) in zip(fdf.gender, fdf.account_type)])
"""